# ============================================================
# STEP 2C — HYBRID SIMILARITY FUSION
# Ontology-Guided Hybrid CAD Assembly Retrieval Framework
# Target: Advanced Engineering Informatics (Elsevier)
#
# Formula:
#   Hybrid Score = 0.6 × technical_similarity
#                + 0.4 × ontology_similarity
#
# This creates:
#   "Ontology-Guided Hybrid Engineering Retrieval"
# ============================================================

import os
import pandas as pd
import numpy as np

os.makedirs("outputs/tables", exist_ok=True)
print("Output folders ready.")

# ============================================================
# LOAD TECHNICAL SIMILARITY MATRIX
# ============================================================

print("\nLoading technical similarity matrix...")
tech_df = pd.read_csv("outputs/tables/technical_similarity_matrix.csv", index_col=0)
tech_df.index.name = 'assembly_name'
print(f"Technical similarity: {tech_df.shape}")

# ============================================================
# LOAD ONTOLOGY SIMILARITY MATRIX
# ============================================================

print("Loading ontology similarity matrix...")
onto_df = pd.read_csv("outputs/tables/ontology_similarity_scores.csv", index_col=0)
onto_df.index.name = 'assembly_name'
print(f"Ontology similarity: {onto_df.shape}")

# ============================================================
# ALIGN INDICES (ensure same assembly order)
# ============================================================

common_assemblies = sorted(set(tech_df.index) & set(onto_df.index))
print(f"\nCommon assemblies: {len(common_assemblies)}")

tech_aligned = tech_df.loc[common_assemblies, common_assemblies]
onto_aligned = onto_df.loc[common_assemblies, common_assemblies]

# ============================================================
# NORMALIZE BOTH MATRICES TO [0, 1]
# ============================================================

def normalize_matrix(mat):
    arr = mat.values.astype(float)
    min_val, max_val = arr.min(), arr.max()
    if max_val - min_val < 1e-10:
        return mat
    normalized = (arr - min_val) / (max_val - min_val)
    return pd.DataFrame(normalized, index=mat.index, columns=mat.columns)

tech_norm = normalize_matrix(tech_aligned)
onto_norm = normalize_matrix(onto_aligned)
print("Both matrices normalized to [0, 1].")

# ============================================================
# HYBRID FUSION: 0.6 × technical + 0.4 × ontology
# ============================================================

ALPHA_TECH = 0.6
ALPHA_ONTO = 0.4

print(f"\nApplying hybrid fusion:")
print(f"  alpha_technical = {ALPHA_TECH}")
print(f"  alpha_ontology  = {ALPHA_ONTO}")

hybrid_arr = (ALPHA_TECH * tech_norm.values) + (ALPHA_ONTO * onto_norm.values)
np.fill_diagonal(hybrid_arr, 1.0)

hybrid_df = pd.DataFrame(hybrid_arr, index=common_assemblies, columns=common_assemblies)
hybrid_df.index.name = 'assembly_name'

hybrid_df.to_csv("outputs/tables/hybrid_similarity_matrix.csv")
print(f"Saved: hybrid_similarity_matrix.csv  shape={hybrid_df.shape}")

# ============================================================
# HYBRID WEIGHTS TABLE
# ============================================================

weights_df = pd.DataFrame([
    {'Component': 'Technical Similarity', 'Weight': ALPHA_TECH,
     'Features': 'mass, volume, edge_count, face_count, vertex_count, surface_area, complexity_ratio, geometric_complexity, component_count, body_count, rotational_symmetry'},
    {'Component': 'Ontology Similarity', 'Weight': ALPHA_ONTO,
     'Features': 'hasMaterial, belongsToFamily, hasTopology, hasComplexityLevel, hasStructure'},
])
weights_df.to_csv("outputs/tables/hybrid_weights.csv", index=False)
print("Saved: hybrid_weights.csv")

# ============================================================
# HYBRID RETRIEVAL RESULTS — TOP-10 NEIGHBORS PER ASSEMBLY
# ============================================================

records = []
for i, name in enumerate(common_assemblies):
    row = hybrid_arr[i].copy()
    row[i] = -1
    top10_idx = np.argsort(row)[::-1][:10]
    for rank, j in enumerate(top10_idx, start=1):
        records.append({
            'query_assembly': name,
            'rank': rank,
            'retrieved_assembly': common_assemblies[j],
            'hybrid_similarity': round(hybrid_arr[i, j], 6),
            'technical_component': round(ALPHA_TECH * tech_norm.values[i, j], 6),
            'ontology_component': round(ALPHA_ONTO * onto_norm.values[i, j], 6),
        })

hybrid_retrieval_df = pd.DataFrame(records)
hybrid_retrieval_df.to_csv("outputs/tables/hybrid_retrieval_results.csv", index=False)
print(f"Saved: hybrid_retrieval_results.csv  ({len(hybrid_retrieval_df)} rows)")

# ============================================================
# TOP NEIGHBORS TABLE (top-5, wide format)
# ============================================================

top_records = []
for i, name in enumerate(common_assemblies):
    row = hybrid_arr[i].copy()
    row[i] = -1
    top5_idx = np.argsort(row)[::-1][:5]
    entry = {'assembly_name': name}
    for rank, j in enumerate(top5_idx, start=1):
        entry[f'neighbor_{rank}'] = common_assemblies[j]
        entry[f'hybrid_sim_{rank}'] = round(hybrid_arr[i, j], 6)
    top_records.append(entry)

pd.DataFrame(top_records).to_csv("outputs/tables/hybrid_top_neighbors.csv", index=False)
print(f"Saved: hybrid_top_neighbors.csv  ({len(top_records)} assemblies)")

# ============================================================
# SIMILARITY STATISTICS
# ============================================================

upper_tri = hybrid_arr[np.triu_indices(len(common_assemblies), k=1)]
stats = {
    'Assemblies': len(common_assemblies),
    'Technical Weight': ALPHA_TECH,
    'Ontology Weight': ALPHA_ONTO,
    'Mean Hybrid Similarity': round(float(np.mean(upper_tri)), 6),
    'Std Hybrid Similarity': round(float(np.std(upper_tri)), 6),
    'Min Hybrid Similarity': round(float(np.min(upper_tri)), 6),
    'Max Hybrid Similarity': round(float(np.max(upper_tri)), 6),
    'Median Hybrid Similarity': round(float(np.median(upper_tri)), 6),
}
pd.DataFrame(list(stats.items()), columns=['Statistic', 'Value']).to_csv(
    "outputs/tables/hybrid_similarity_statistics.csv", index=False)

print("\n--- Hybrid Similarity Statistics ---")
for k, v in stats.items():
    print(f"  {k}: {v}")

print("\n✅ STEP 2C COMPLETE — Hybrid Similarity Fusion Done")
print(f"   Formula: {ALPHA_TECH} x technical + {ALPHA_ONTO} x ontology")
