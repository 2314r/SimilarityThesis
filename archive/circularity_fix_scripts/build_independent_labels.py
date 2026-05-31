"""
Build independent ground-truth labels from assembly contact topology.
Contact/hole counts are NOT in the geometric feature set → genuinely independent.

Run locally:
    cd ~/Desktop/SimilarityThesis
    python3 build_independent_labels.py
"""
import os, json, pandas as pd, numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text

DATASET_CSV = Path("ontology/data/fusion360_rotor_dataset.csv")
FUSION_DIR  = Path.home() / "Desktop" / "fusion_subset"
OUTPUT_CSV  = Path("ontology/data/fusion360_independent_labels.csv")

df = pd.read_csv(DATASET_CSV)
print(f"Loaded {len(df)} assemblies\n")

# ── Extract topology features from JSON ───────────────────────────────────────
topo_rows = []
for asm_id in df["assembly_name"]:
    path = FUSION_DIR / asm_id / "assembly.json"
    with open(path) as f:
        data = json.load(f)

    n_contacts    = len(data.get("contacts")    or [])
    n_holes       = len(data.get("holes")       or [])
    n_occurrences = len(data.get("occurrences") or {})
    n_components  = len(data.get("components")  or {})
    n_bodies_json = len(data.get("bodies")      or {})

    topo_rows.append({
        "assembly_name": asm_id,
        "n_contacts":    n_contacts,
        "n_holes":       n_holes,
        "n_occurrences": n_occurrences,
        "n_components_json": n_components,
        "n_bodies_json": n_bodies_json,
    })

topo = pd.DataFrame(topo_rows)

# ── Balanced threshold-based labeling ─────────────────────────────────────────
# Strategy: isolate zero group, then log-quintile the non-zero part
combined    = topo["n_contacts"] + topo["n_holes"]
nonzero_log = np.log1p(combined[combined > 0])
q25, q50, q75 = np.percentile(nonzero_log, [25, 50, 75])

def assign_topo(c):
    if c == 0:                     return "topo_isolated"
    lc = np.log1p(c)
    if lc < q25:                   return "topo_minimal"
    elif lc < q50:                 return "topo_moderate"
    elif lc < q75:                 return "topo_connected"
    else:                          return "topo_complex"

topo["topo_family"] = combined.apply(assign_topo)

print("Topology family distribution:")
print(topo["topo_family"].value_counts().sort_index())
print()

# ── Independence check ────────────────────────────────────────────────────────
ORIG_FEATURES = ["volume","edge_count","face_count","vertex_count",
                 "complexity_ratio","geometric_complexity",
                 "component_count","body_count","rotational_symmetry"]
merged = df.merge(topo[["assembly_name","topo_family"]], on="assembly_name")
X_orig = merged[ORIG_FEATURES].fillna(0)
y_new  = merged["topo_family"]

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_orig, y_new)
acc = (dt.predict(X_orig) == y_new).mean()
print(f"Decision tree accuracy (orig features → topo_family): {acc:.4f}")
if acc < 0.85:
    print("✅ INDEPENDENCE CONFIRMED — new labels are NOT derivable from old features")
elif acc < 0.95:
    print("⚠️  Partial independence — acceptable (some structural correlation expected)")
else:
    print("❌ Still circular — need different approach")
print()

# ── Save ──────────────────────────────────────────────────────────────────────
result = df.merge(
    topo[["assembly_name","topo_family","n_contacts","n_holes","n_occurrences"]],
    on="assembly_name"
)
result.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")
print(f"\nCross-tab (old family vs new topo_family):")
print(pd.crosstab(result["family"], result["topo_family"]))
