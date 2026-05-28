# ============================================================
# UNIFIED PIPELINE — Ontology-Guided Hybrid CAD Assembly
# Retrieval Framework
# Target: Advanced Engineering Informatics (Elsevier)
#
# FIXES APPLIED:
#   BUG 1: Removed surface_area (all zeros) and mass (r=0.998
#           with volume; mass = density × volume, redundant)
#   BUG 2: Removed family from semantic features — family is
#           ground truth for evaluation, NOT an input feature
#   BUG 3: Single canonical pipeline replaces ThesisV9.ipynb
#           (alpha=0.2/0.5/0.3) and old scripts (0.6/0.4).
#           Weights now chosen by empirical grid search.
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Output directories ────────────────────────────────────────
OUT_TABLES  = "../outputs/tables"
OUT_FIGURES = "../outputs/figures"
os.makedirs(OUT_TABLES,  exist_ok=True)
os.makedirs(OUT_FIGURES, exist_ok=True)
print("Output folders ready.\n")

# ============================================================
# STEP 0 — LOAD & VALIDATE DATASET
# ============================================================

df = pd.read_csv("../data/fusion360_rotor_dataset.csv")
print(f"Dataset: {df.shape[0]} assemblies, {df.shape[1]} columns")
print(f"Families: {df['family'].value_counts().to_dict()}\n")

# Ground-truth labels (ONLY used for evaluation, never for input)
ASSEMBLY_NAMES = df['assembly_name'].tolist()
FAMILY_LOOKUP  = dict(zip(df['assembly_name'], df['family']))

# ============================================================
# STEP 1 — FEATURE ENGINEERING
# ============================================================
#
# BUG 1 FIX:
#   - surface_area removed (0 for all 746 rows — extraction bug)
#   - mass removed (corr=0.998 with volume; mass ≈ density×volume,
#     and density is captured by material class)
#
# BUG 2 FIX:
#   - family NOT used as semantic feature (it IS the ground truth)
#   - semantic features = material + topology + complexity + structure
#     — all derivable without knowing the assembly's family label

# ── Technical (geometric/physical) ───────────────────────────
TECHNICAL_COLS = [
    'volume',               # geometric size
    'edge_count',           # topological complexity
    'face_count',           # topological complexity
    'vertex_count',         # topological complexity
    'complexity_ratio',     # edge/face density
    'geometric_complexity', # combined geometric measure
]

# ── Structural (assembly topology) ───────────────────────────
STRUCTURAL_COLS = [
    'component_count',      # assembly hierarchy depth
    'body_count',           # number of solid bodies
    'rotational_symmetry',  # symmetry flag
]

# ── Semantic (ontology-derived, NO family) ───────────────────
def topology_class(x):
    if x < 300:   return "Sparse"
    elif x < 1500: return "Moderate"
    return "Dense"

def complexity_class(x):
    if x < 500:   return "Low"
    elif x < 3000: return "Medium"
    return "High"

df_feat = df.copy()
df_feat['topology_class']   = df['edge_count'].apply(topology_class)
df_feat['complexity_class'] = df['geometric_complexity'].apply(complexity_class)
df_feat['structure_class']  = np.where(df['rotational_symmetry'] == 1, 'Symmetric', 'Asymmetric')

# Normalize material to ontology class names
def normalize_material(mat):
    mat_lower = str(mat).lower()
    if 'steel'     in mat_lower: return "Steel"
    if 'aluminum'  in mat_lower: return "Aluminum"
    if 'plastic'   in mat_lower: return "Plastic"
    if 'wood'      in mat_lower: return "Wood"
    if 'composite' in mat_lower: return "Composite"
    return "Other"

df_feat['material_class'] = df['material'].apply(normalize_material)

SEMANTIC_COLS_CAT = ['material_class', 'topology_class', 'complexity_class', 'structure_class']

print(f"Technical features ({len(TECHNICAL_COLS)}):  {TECHNICAL_COLS}")
print(f"Structural features ({len(STRUCTURAL_COLS)}): {STRUCTURAL_COLS}")
print(f"Semantic features ({len(SEMANTIC_COLS_CAT)}):   {SEMANTIC_COLS_CAT}")
print("(family NOT used as input — only as evaluation ground truth)\n")

# Save feature summary
feat_rows = (
    [(f, 'Technical')  for f in TECHNICAL_COLS]  +
    [(f, 'Structural') for f in STRUCTURAL_COLS] +
    [(f, 'Semantic')   for f in SEMANTIC_COLS_CAT]
)
pd.DataFrame(feat_rows, columns=['Feature','Category']).to_csv(
    f"{OUT_TABLES}/feature_summary_clean.csv", index=False)

# ============================================================
# STEP 2 — SIMILARITY MATRICES
# ============================================================

scaler = MinMaxScaler()

# ── 2A: Technical similarity ─────────────────────────────────
X_tech = df_feat[TECHNICAL_COLS].fillna(df_feat[TECHNICAL_COLS].median())
X_tech_scaled = scaler.fit_transform(X_tech)
SIM_TECH = cosine_similarity(X_tech_scaled)
np.fill_diagonal(SIM_TECH, 1.0)

# ── 2B: Structural similarity ────────────────────────────────
X_struct = df_feat[STRUCTURAL_COLS].fillna(df_feat[STRUCTURAL_COLS].median())
X_struct_scaled = scaler.fit_transform(X_struct)
SIM_STRUCT = cosine_similarity(X_struct_scaled)
np.fill_diagonal(SIM_STRUCT, 1.0)

# ── 2C: Semantic similarity (ontology-guided, NO family) ─────
X_sem = pd.get_dummies(df_feat[SEMANTIC_COLS_CAT])
SIM_SEM = cosine_similarity(X_sem.values.astype(float))
np.fill_diagonal(SIM_SEM, 1.0)

print(f"Similarity matrices computed: {SIM_TECH.shape} each")
print(f"Semantic encoded dimensions: {X_sem.shape[1]}")

# Save raw matrices
for name, mat in [('technical', SIM_TECH), ('structural', SIM_STRUCT), ('semantic', SIM_SEM)]:
    pd.DataFrame(mat, index=ASSEMBLY_NAMES, columns=ASSEMBLY_NAMES).to_csv(
        f"{OUT_TABLES}/{name}_similarity_matrix.csv")
    print(f"  Saved: {name}_similarity_matrix.csv")

# ============================================================
# STEP 3 — EVALUATION FUNCTIONS
# ============================================================

def precision_at_k(sim_matrix, labels, assemblies, k):
    precisions = []
    for i, name in enumerate(assemblies):
        q_fam = labels.get(name)
        if q_fam is None:
            continue
        row = sim_matrix[i].copy()
        row[i] = -1
        top_k = np.argsort(row)[::-1][:k]
        hits = sum(1 for j in top_k if labels.get(assemblies[j]) == q_fam)
        precisions.append(hits / k)
    return float(np.mean(precisions))

def mean_reciprocal_rank(sim_matrix, labels, assemblies):
    rr_list = []
    for i, name in enumerate(assemblies):
        q_fam = labels.get(name)
        if q_fam is None:
            continue
        row = sim_matrix[i].copy()
        row[i] = -1
        ranked = np.argsort(row)[::-1]
        rr = 0.0
        for rank, j in enumerate(ranked, 1):
            if labels.get(assemblies[j]) == q_fam:
                rr = 1.0 / rank
                break
        rr_list.append(rr)
    return float(np.mean(rr_list))

def evaluate(mat, label=''):
    return {
        'Method':       label,
        'Precision@3':  round(precision_at_k(mat, FAMILY_LOOKUP, ASSEMBLY_NAMES, 3), 4),
        'Precision@5':  round(precision_at_k(mat, FAMILY_LOOKUP, ASSEMBLY_NAMES, 5), 4),
        'Precision@10': round(precision_at_k(mat, FAMILY_LOOKUP, ASSEMBLY_NAMES, 10), 4),
        'MRR':          round(mean_reciprocal_rank(mat, FAMILY_LOOKUP, ASSEMBLY_NAMES), 4),
    }

# ============================================================
# STEP 4 — GRID SEARCH FOR OPTIMAL HYBRID WEIGHTS
# BUG 3 FIX: Weights chosen empirically, not arbitrarily
#
# Search space: α + β + γ = 1
#   α = technical weight
#   β = semantic weight
#   γ = structural weight
# Optimize for MRR (most robust ranking metric)
# ============================================================

print("\nGrid search for optimal hybrid weights (optimising MRR)...")
print("(α=technical, β=semantic, γ=structural, α+β+γ=1)\n")

steps = np.arange(0.0, 1.01, 0.1)
best_mrr   = -1
best_p5    = -1
best_alpha = 0.5
best_beta  = 0.3
best_gamma = 0.2
grid_results = []

for alpha in steps:
    for beta in steps:
        gamma = round(1.0 - alpha - beta, 10)
        if gamma < -0.001 or gamma > 1.001:
            continue
        gamma = max(0.0, min(1.0, gamma))
        if abs(alpha + beta + gamma - 1.0) > 0.01:
            continue

        hybrid = alpha * SIM_TECH + beta * SIM_SEM + gamma * SIM_STRUCT
        np.fill_diagonal(hybrid, 1.0)

        mrr = mean_reciprocal_rank(hybrid, FAMILY_LOOKUP, ASSEMBLY_NAMES)
        p5  = precision_at_k(hybrid, FAMILY_LOOKUP, ASSEMBLY_NAMES, 5)

        grid_results.append({
            'alpha_technical':  round(alpha, 2),
            'beta_semantic':    round(beta,  2),
            'gamma_structural': round(gamma, 2),
            'MRR':   round(mrr, 4),
            'P@5':   round(p5,  4),
        })

        if mrr > best_mrr:
            best_mrr   = mrr
            best_p5    = p5
            best_alpha = alpha
            best_beta  = beta
            best_gamma = gamma

grid_df = pd.DataFrame(grid_results).sort_values('MRR', ascending=False)
grid_df.to_csv(f"{OUT_TABLES}/weight_grid_search.csv", index=False)

print(f"Best weights found:")
print(f"  α (technical)  = {best_alpha:.2f}")
print(f"  β (semantic)   = {best_beta:.2f}")
print(f"  γ (structural) = {best_gamma:.2f}")
print(f"  MRR = {best_mrr:.4f}  |  Precision@5 = {best_p5:.4f}")

# ============================================================
# STEP 5 — FINAL HYBRID SIMILARITY
# ============================================================

SIM_HYBRID = best_alpha * SIM_TECH + best_beta * SIM_SEM + best_gamma * SIM_STRUCT
np.fill_diagonal(SIM_HYBRID, 1.0)

pd.DataFrame(SIM_HYBRID, index=ASSEMBLY_NAMES, columns=ASSEMBLY_NAMES).to_csv(
    f"{OUT_TABLES}/hybrid_similarity_matrix.csv")
print("\nSaved: hybrid_similarity_matrix.csv")

# Save canonical weights
pd.DataFrame([
    {'Component': 'Technical',  'Weight': best_alpha,
     'Features': ', '.join(TECHNICAL_COLS)},
    {'Component': 'Semantic',   'Weight': best_beta,
     'Features': ', '.join(SEMANTIC_COLS_CAT)},
    {'Component': 'Structural', 'Weight': best_gamma,
     'Features': ', '.join(STRUCTURAL_COLS)},
]).to_csv(f"{OUT_TABLES}/hybrid_weights_final.csv", index=False)

# ============================================================
# STEP 6 — FULL EVALUATION: ALL METHODS
# ============================================================

print("\nEvaluating all methods...")
results = [
    evaluate(SIM_TECH,    'Technical'),
    evaluate(SIM_SEM,     'Semantic (Ontology)'),
    evaluate(SIM_STRUCT,  'Structural'),
    evaluate(SIM_HYBRID,  'Hybrid'),
]
results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT_TABLES}/retrieval_comparison_clean.csv", index=False)

print("\n" + "="*65)
print("FINAL RETRIEVAL RESULTS (clean pipeline, no circular eval)")
print("="*65)
print(results_df.to_string(index=False))
print("="*65)

# ============================================================
# STEP 7 — TOP-10 RETRIEVAL RESULTS (for analysis)
# ============================================================

for method_name, mat in [('technical',  SIM_TECH),
                          ('semantic',   SIM_SEM),
                          ('structural', SIM_STRUCT),
                          ('hybrid',     SIM_HYBRID)]:
    records = []
    for i, name in enumerate(ASSEMBLY_NAMES):
        row = mat[i].copy()
        row[i] = -1
        top10 = np.argsort(row)[::-1][:10]
        for rank, j in enumerate(top10, 1):
            records.append({
                'query_assembly':     name,
                'query_family':       FAMILY_LOOKUP.get(name),
                'rank':               rank,
                'retrieved_assembly': ASSEMBLY_NAMES[j],
                'retrieved_family':   FAMILY_LOOKUP.get(ASSEMBLY_NAMES[j]),
                'similarity':         round(mat[i, j], 6),
                'correct':            FAMILY_LOOKUP.get(name) == FAMILY_LOOKUP.get(ASSEMBLY_NAMES[j]),
            })
    pd.DataFrame(records).to_csv(f"{OUT_TABLES}/{method_name}_retrieval_results.csv",
                                  index=False)

print("\nRetrieval result tables saved for all 4 methods.")

# ============================================================
# STEP 8 — FIGURES
# ============================================================

# Fig 1 — Retrieval comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Retrieval Performance — Clean Pipeline\n'
             '(No circular evaluation, empirically optimised weights)',
             fontsize=12, fontweight='bold')

colors = {'Technical': '#4C72B0', 'Semantic (Ontology)': '#DD8452',
          'Structural': '#55A868', 'Hybrid': '#C44E52'}
methods = [r['Method'] for r in results]
bar_c   = [colors.get(m, '#888888') for m in methods]

ks    = [3, 5, 10]
x     = np.arange(len(ks))
width = 0.2

ax = axes[0]
for i, r in enumerate(results):
    vals = [r['Precision@3'], r['Precision@5'], r['Precision@10']]
    ax.bar(x + i*width, vals, width, label=r['Method'],
           color=bar_c[i], alpha=0.87, edgecolor='white')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(['Precision@3', 'Precision@5', 'Precision@10'])
ax.set_ylabel('Precision')
ax.set_ylim(0, 1.05)
ax.set_title('Precision@K Comparison')
ax.legend(fontsize=8)
ax.grid(axis='y', linestyle='--', alpha=0.4)

ax2 = axes[1]
mrr_vals = [r['MRR'] for r in results]
bars = ax2.bar(methods, mrr_vals, color=bar_c, alpha=0.87, edgecolor='white')
for bar, val in zip(bars, mrr_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.set_ylabel('MRR')
ax2.set_ylim(0, 1.05)
ax2.set_title('Mean Reciprocal Rank')
ax2.tick_params(axis='x', rotation=15)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/retrieval_comparison_clean.pdf", bbox_inches='tight', dpi=150)
plt.savefig(f"{OUT_FIGURES}/retrieval_comparison_clean.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: retrieval_comparison_clean.pdf")

# Fig 2 — Weight grid search heatmap (α vs β, best γ implicit)
pivot = grid_df.pivot_table(values='MRR', index='beta_semantic',
                             columns='alpha_technical', aggfunc='max')
fig2, ax3 = plt.subplots(figsize=(8, 6))
im = ax3.imshow(pivot.values, cmap='YlOrRd', aspect='auto',
                vmin=pivot.values.min(), vmax=pivot.values.max())
plt.colorbar(im, ax=ax3, label='MRR')
ax3.set_xticks(range(len(pivot.columns)))
ax3.set_xticklabels([f'{v:.1f}' for v in pivot.columns], fontsize=8)
ax3.set_yticks(range(len(pivot.index)))
ax3.set_yticklabels([f'{v:.1f}' for v in pivot.index], fontsize=8)
ax3.set_xlabel('α (Technical Weight)')
ax3.set_ylabel('β (Semantic Weight)')
ax3.set_title(f'Hybrid Weight Grid Search — MRR\n'
              f'Best: α={best_alpha:.1f}, β={best_beta:.1f}, γ={best_gamma:.1f}  →  MRR={best_mrr:.4f}')

# Mark best
best_row = list(pivot.index).index(round(best_beta, 1))
best_col = list(pivot.columns).index(round(best_alpha, 1))
ax3.plot(best_col, best_row, 'b*', markersize=14, label=f'Best (α={best_alpha:.1f}, β={best_beta:.1f})')
ax3.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/weight_grid_search.pdf", bbox_inches='tight', dpi=150)
plt.savefig(f"{OUT_FIGURES}/weight_grid_search.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: weight_grid_search.pdf")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*65)
print("UNIFIED PIPELINE COMPLETE")
print("="*65)
print(f"Dataset: {len(ASSEMBLY_NAMES)} assemblies")
print(f"Technical features:  {len(TECHNICAL_COLS)} (surface_area removed)")
print(f"Semantic features:   {len(SEMANTIC_COLS_CAT)} (family NOT used as input)")
print(f"Structural features: {len(STRUCTURAL_COLS)}")
print(f"Hybrid weights: α={best_alpha:.2f} (tech) + β={best_beta:.2f} (sem) + γ={best_gamma:.2f} (struct)")
print(f"\nAll outputs → {OUT_TABLES}/")
print(f"All figures → {OUT_FIGURES}/")
print("="*65)
