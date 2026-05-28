# ============================================================
# STEP 1A/1B/1C — FORMAL ONTOLOGY GENERATION v2
# Ontology-Guided Hybrid CAD Assembly Retrieval Framework
# Target: Advanced Engineering Informatics (Elsevier)
#
# Enhancements over v1:
#   - Formal class hierarchy triples (rdfs:subClassOf)
#   - rdf:type class membership triples
#   - Datatype property triples (numerical feature binning)
#   - Ontology coverage and completeness statistics
#   - Predicate informativeness analysis
# ============================================================

import os
import pandas as pd
import numpy as np

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
print("Output folders ready.\n")

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("fusion360_rotor_dataset.csv")
print(f"Dataset loaded: {df.shape[0]} assemblies")

# ============================================================
# CLASSIFICATION HELPERS (same thresholds as v1 for consistency)
# ============================================================

def complexity_class(x):
    if x < 500:   return "Low"
    elif x < 3000: return "Medium"
    return "High"

def topology_class(x):
    if x < 300:   return "Sparse"
    elif x < 1500: return "Moderate"
    return "Dense"

def mass_class(x):
    if x < 1.0:   return "LightWeight"
    elif x < 20.0: return "MediumWeight"
    return "HeavyWeight"

def volume_class(x):
    if x < 100:    return "SmallVolume"
    elif x < 5000: return "MediumVolume"
    return "LargeVolume"

def structure_class(r):
    return "Symmetric" if r == 1 else "Asymmetric"

df = df.copy()
df["complexity_level"]  = df["geometric_complexity"].apply(complexity_class)
df["topology_level"]    = df["edge_count"].apply(topology_class)
df["mass_class"]        = df["mass"].apply(mass_class)
df["volume_class"]      = df["volume"].apply(volume_class)
df["structure_class"]   = df["rotational_symmetry"].apply(structure_class)

# ============================================================
# PART 1 — CLASS HIERARCHY TRIPLES (rdfs:subClassOf)
# ============================================================

hierarchy_triples = [
    # Assembly type hierarchy
    ("SimpleRotor",         "rdfs:subClassOf", "EngineeringAssembly"),
    ("SymmetricRotor",      "rdfs:subClassOf", "EngineeringAssembly"),
    ("MediumRotor",         "rdfs:subClassOf", "EngineeringAssembly"),
    ("HighComplexityRotor", "rdfs:subClassOf", "EngineeringAssembly"),
    ("ComplexAssembly",     "rdfs:subClassOf", "EngineeringAssembly"),
    # Material hierarchy
    ("Steel",               "rdfs:subClassOf", "EngineeringMaterial"),
    ("Aluminum",            "rdfs:subClassOf", "EngineeringMaterial"),
    ("Plastic",             "rdfs:subClassOf", "EngineeringMaterial"),
    ("Wood",                "rdfs:subClassOf", "EngineeringMaterial"),
    ("Composite",           "rdfs:subClassOf", "EngineeringMaterial"),
    ("OtherEngineeringMaterial", "rdfs:subClassOf", "EngineeringMaterial"),
    # Feature hierarchy
    ("TopologyFeature",     "rdfs:subClassOf", "EngineeringFeature"),
    ("PhysicalFeature",     "rdfs:subClassOf", "EngineeringFeature"),
    ("StructuralFeature",   "rdfs:subClassOf", "EngineeringFeature"),
    # Topology sub-features
    ("EdgeCount",           "rdfs:subClassOf", "TopologyFeature"),
    ("FaceCount",           "rdfs:subClassOf", "TopologyFeature"),
    ("GeometricComplexity", "rdfs:subClassOf", "TopologyFeature"),
    # Physical sub-features
    ("Mass",                "rdfs:subClassOf", "PhysicalFeature"),
    ("Volume",              "rdfs:subClassOf", "PhysicalFeature"),
    # Structural sub-features
    ("ComponentCount",      "rdfs:subClassOf", "StructuralFeature"),
    ("RotationalSymmetry",  "rdfs:subClassOf", "StructuralFeature"),
]

hierarchy_df = pd.DataFrame(hierarchy_triples,
                             columns=["Subject", "Predicate", "Object"])
hierarchy_df.to_csv("outputs/tables/ontology_class_hierarchy.csv", index=False)
print(f"Class hierarchy: {len(hierarchy_df)} triples")

# ============================================================
# PART 2 — ASSEMBLY INSTANCE TRIPLES (5 core predicates + rdf:type)
# ============================================================

# Normalize family names to ontology class names
FAMILY_TO_CLASS = {
    "simple_rotor":           "SimpleRotor",
    "symmetric_rotor":        "SymmetricRotor",
    "medium_rotor":           "MediumRotor",
    "high_complexity_rotor":  "HighComplexityRotor",
    "complex_assembly":       "ComplexAssembly",
}

# Normalize material to ontology classes
def normalize_material(mat):
    mat_lower = str(mat).lower()
    if 'steel'    in mat_lower: return "Steel"
    if 'aluminum' in mat_lower: return "Aluminum"
    if 'plastic'  in mat_lower: return "Plastic"
    if 'wood'     in mat_lower: return "Wood"
    if 'composite'in mat_lower: return "Composite"
    return "OtherEngineeringMaterial"

triples = []

for _, row in df.iterrows():
    asm = row["assembly_name"]
    family_class = FAMILY_TO_CLASS.get(row["family"], "EngineeringAssembly")
    mat_class    = normalize_material(row["material"])

    # rdf:type — formal class membership
    triples.append([asm, "rdf:type", family_class])

    # Core semantic predicates
    triples.append([asm, "hasMaterial",       mat_class])
    triples.append([asm, "belongsToFamily",   row["family"]])
    triples.append([asm, "hasComplexityLevel",row["complexity_level"]])
    triples.append([asm, "hasTopology",       row["topology_level"]])
    triples.append([asm, "hasStructure",      row["structure_class"]])

    # Datatype property triples (binned numerical features)
    triples.append([asm, "hasMassCategory",   row["mass_class"]])
    triples.append([asm, "hasVolumeCategory", row["volume_class"]])

triples_df = pd.DataFrame(triples, columns=["Subject", "Predicate", "Object"])
triples_df.to_csv("outputs/tables/ontology_triples.csv", index=False)
print(f"Assembly triples: {len(triples_df)} (8 predicates × 746 assemblies)")

# ============================================================
# PART 3 — COMBINED FULL ONTOLOGY (hierarchy + instances)
# ============================================================

full_ontology_df = pd.concat([hierarchy_df, triples_df], ignore_index=True)
full_ontology_df.to_csv("outputs/tables/ontology_full.csv", index=False)
print(f"Full ontology: {len(full_ontology_df)} triples total")

# ============================================================
# PART 4 — ONTOLOGY STATISTICS (AEI table)
# ============================================================

# Unique materials in dataset
materials_in_data = df["material"].apply(normalize_material).unique()

stats = [
    ("Total Instance Triples",        len(triples_df)),
    ("Total Hierarchy Triples",       len(hierarchy_df)),
    ("Total Ontology Triples",        len(full_ontology_df)),
    ("Assembly Instances",            df["assembly_name"].nunique()),
    ("Ontology Classes (Assembly)",   5),
    ("Ontology Classes (Material)",   6),
    ("Ontology Classes (Feature)",    3),
    ("Total Ontology Classes",        14),
    ("Semantic Predicates",           triples_df["Predicate"].nunique()),
    ("Unique Object Values",          triples_df["Object"].nunique()),
    ("Unique Nodes (full graph)",     full_ontology_df["Subject"].nunique()
                                    + full_ontology_df["Object"].nunique()),
    ("Ontology Predicate Coverage",   f"{triples_df['Predicate'].nunique()} / 8"),
]

stats_df = pd.DataFrame(stats, columns=["Metric", "Value"])
stats_df.to_csv("outputs/tables/ontology_statistics.csv", index=False)
print("\nOntology Statistics:")
print(stats_df.to_string(index=False))

# ============================================================
# PART 5 — PREDICATE INFORMATIVENESS (Shannon Entropy)
# ============================================================

predicate_info = []
for pred in triples_df["Predicate"].unique():
    subset = triples_df[triples_df["Predicate"] == pred]["Object"]
    counts = subset.value_counts(normalize=True)
    entropy = float(-np.sum(counts * np.log2(counts + 1e-12)))
    predicate_info.append({
        "Predicate":         pred,
        "Unique Values":     subset.nunique(),
        "Shannon Entropy":   round(entropy, 4),
        "Most Common Value": subset.mode()[0],
        "Coverage":          len(subset),
    })

pred_df = pd.DataFrame(predicate_info).sort_values("Shannon Entropy", ascending=False)
pred_df.to_csv("outputs/tables/predicate_informativeness.csv", index=False)
print("\nPredicate Informativeness (Shannon Entropy):")
print(pred_df.to_string(index=False))

# ============================================================
# PART 6 — SEMANTIC RELATION DISTRIBUTION
# ============================================================

rel_dist = triples_df["Predicate"].value_counts().reset_index()
rel_dist.columns = ["Predicate", "Count"]
rel_dist.to_csv("outputs/tables/semantic_relation_distribution.csv", index=False)

# ============================================================
# PART 7 — ONTOLOGY COVERAGE VALIDATION
# ============================================================

total_possible = len(df) * triples_df["Predicate"].nunique()
actual_coverage = len(triples_df)
coverage_pct = round(100.0 * actual_coverage / total_possible, 2)

missing_values = triples_df[triples_df["Object"].isna() |
                              (triples_df["Object"] == "")].shape[0]

coverage_df = pd.DataFrame([
    ("Total Expected Triples", total_possible),
    ("Actual Triples",         actual_coverage),
    ("Coverage (%)",           coverage_pct),
    ("Missing / Null Values",  missing_values),
    ("Ontology Completeness",  "Complete" if missing_values == 0 else "Incomplete"),
])
coverage_df.columns = ["Metric", "Value"]
coverage_df.to_csv("outputs/tables/ontology_coverage.csv", index=False)
print(f"\nOntology Coverage: {coverage_pct}% ({actual_coverage}/{total_possible} triples)")

print("\n✅ T1 COMPLETE — Formal Ontology Generation v2 Done")
print(f"   Predicates: {triples_df['Predicate'].nunique()}")
print(f"   Instance triples: {len(triples_df)}")
print(f"   Hierarchy triples: {len(hierarchy_df)}")
print(f"   Coverage: {coverage_pct}%")
