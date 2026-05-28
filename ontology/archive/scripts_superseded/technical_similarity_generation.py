# ============================================================
# STEP 2B.5 — TECHNICAL SIMILARITY MATRIX GENERATION
# Ontology-Guided Hybrid CAD Assembly Retrieval Framework
# Target: Advanced Engineering Informatics (Elsevier)
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs("outputs/tables", exist_ok=True)
print("Output folders ready.")

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("fusion360_rotor_dataset.csv")
print(f"\nDataset loaded: {df.shape[0]} assemblies, {df.shape[1]} columns")

# ============================================================
# TECHNICAL FEATURE GROUPS (as per project spec)
# ============================================================

# Technical features: physical/geometric properties
technical_cols = ['mass', 'volume', 'edge_count', 'face_count',
                  'vertex_count', 'surface_area', 'complexity_ratio',
                  'geometric_complexity']

# Structural features: component topology
structural_cols = ['component_count', 'body_count', 'rotational_symmetry']

# Combined technical + structural = full technical representation
feature_cols = [c for c in technical_cols + structural_cols if c in df.columns]

print(f"\nUsing {len(feature_cols)} technical+structural features: {feature_cols}")

# ============================================================
# HANDLE MISSING VALUES
# ============================================================

df_feat = df[['assembly_name'] + feature_cols].copy()
df_feat[feature_cols] = df_feat[feature_cols].fillna(df_feat[feature_cols].median())

# ============================================================
# NORMALIZE FEATURES (MinMax)
# ============================================================

scaler = MinMaxScaler()
X = scaler.fit_transform(df_feat[feature_cols].values)
assembly_names = df_feat['assembly_name'].tolist()

print(f"\nFeature matrix shape: {X.shape}")
print("Normalization complete.")

# ============================================================
# COMPUTE COSINE SIMILARITY MATRIX (746 × 746)
# ============================================================

print("\nComputing cosine similarity matrix (746 × 746)...")
cos_sim = cosine_similarity(X)

# Ensure diagonal = 1.0 (self-similarity)
np.fill_diagonal(cos_sim, 1.0)

# Save as DataFrame with assembly names as index/columns
tech_sim_df = pd.DataFrame(cos_sim, index=assembly_names, columns=assembly_names)
tech_sim_df.index.name = 'assembly_name'

tech_sim_df.to_csv("outputs/tables/technical_similarity_matrix.csv")
print(f"Saved: technical_similarity_matrix.csv  shape={tech_sim_df.shape}")

# ============================================================
# RETRIEVAL RESULTS — TOP-10 NEIGHBORS PER ASSEMBLY
# ============================================================

records = []
for i, name in enumerate(assembly_names):
    row = cos_sim[i].copy()
    row[i] = -1  # exclude self
    top10_idx = np.argsort(row)[::-1][:10]
    for rank, j in enumerate(top10_idx, start=1):
        records.append({
            'query_assembly': name,
            'rank': rank,
            'retrieved_assembly': assembly_names[j],
            'technical_similarity': round(cos_sim[i, j], 6)
        })

retrieval_df = pd.DataFrame(records)
retrieval_df.to_csv("outputs/tables/technical_retrieval_results.csv", index=False)
print(f"Saved: technical_retrieval_results.csv  ({len(retrieval_df)} rows)")

# ============================================================
# TOP NEIGHBORS TABLE (top-5 per assembly, wide format)
# ============================================================

top_neighbor_records = []
for i, name in enumerate(assembly_names):
    row = cos_sim[i].copy()
    row[i] = -1
    top5_idx = np.argsort(row)[::-1][:5]
    entry = {'assembly_name': name}
    for rank, j in enumerate(top5_idx, start=1):
        entry[f'neighbor_{rank}'] = assembly_names[j]
        entry[f'sim_{rank}'] = round(cos_sim[i, j], 6)
    top_neighbor_records.append(entry)

top_neighbors_df = pd.DataFrame(top_neighbor_records)
top_neighbors_df.to_csv("outputs/tables/technical_top_neighbors.csv", index=False)
print(f"Saved: technical_top_neighbors.csv  ({len(top_neighbors_df)} assemblies)")

# ============================================================
# SIMILARITY STATISTICS
# ============================================================

# Flatten upper triangle (exclude diagonal)
upper_tri = cos_sim[np.triu_indices(len(assembly_names), k=1)]

stats = {
    'Assemblies': len(assembly_names),
    'Technical Features Used': len(feature_cols),
    'Mean Similarity': round(float(np.mean(upper_tri)), 6),
    'Std Similarity': round(float(np.std(upper_tri)), 6),
    'Min Similarity': round(float(np.min(upper_tri)), 6),
    'Max Similarity': round(float(np.max(upper_tri)), 6),
    'Median Similarity': round(float(np.median(upper_tri)), 6),
}

stats_df = pd.DataFrame(list(stats.items()), columns=['Statistic', 'Value'])
stats_df.to_csv("outputs/tables/technical_similarity_statistics.csv", index=False)

print("\n--- Technical Similarity Statistics ---")
print(stats_df.to_string(index=False))

print("\n✅ STEP 2B.5 COMPLETE — Technical Similarity Matrix Generated")
