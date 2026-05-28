# ============================================================
# SEMANTIC ONTOLOGY SIMILARITY GENERATION
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs("outputs/tables", exist_ok=True)

print("Output folders ready.")

# ============================================================
# LOAD DATASET
# ============================================================

dataset_path = "../dataset/fusion360_rotor_dataset.csv"

df = pd.read_csv(dataset_path)

print("\nDataset loaded.")
print("Shape:", df.shape)

# ============================================================
# SELECT SEMANTIC FEATURES
# ============================================================

semantic_df = df[[

    "material",
    "family"

]].copy()

# ------------------------------------------------------------
# ADD TOPOLOGY CLASS
# ------------------------------------------------------------

def topology_class(x):

    if x < 300:
        return "Sparse"

    elif x < 1500:
        return "Moderate"

    return "Dense"

semantic_df["topology"] = df["edge_count"].apply(
    topology_class
)

# ------------------------------------------------------------
# ADD COMPLEXITY CLASS
# ------------------------------------------------------------

def complexity_class(x):

    if x < 500:
        return "Low"

    elif x < 3000:
        return "Medium"

    return "High"

semantic_df["complexity"] = df[
    "geometric_complexity"
].apply(complexity_class)

# ------------------------------------------------------------
# ADD STRUCTURE CLASS
# ------------------------------------------------------------

semantic_df["structure"] = np.where(

    df["rotational_symmetry"] == 0,

    "Asymmetric",

    "Symmetric"

)

print("\nSemantic Features Preview:\n")
print(semantic_df.head())

# ============================================================
# ONE-HOT ENCODING
# ============================================================

semantic_encoded = pd.get_dummies(
    semantic_df
)

# ============================================================
# COMPUTE SEMANTIC SIMILARITY
# ============================================================

semantic_similarity = cosine_similarity(
    semantic_encoded
)

semantic_similarity_df = pd.DataFrame(

    semantic_similarity,

    index=df["assembly_name"],

    columns=df["assembly_name"]

)

# ============================================================
# SAVE SIMILARITY MATRIX
# ============================================================

semantic_similarity_df.to_csv(

    "outputs/tables/semantic_similarity_matrix.csv"

)

print("\nSemantic similarity matrix saved.")

# ============================================================
# OVERLAP STATISTICS
# ============================================================

overlap_stats = pd.DataFrame({

    "Metric": [

        "Assemblies",
        "Semantic Features",
        "Encoded Dimensions",
        "Mean Similarity",
        "Max Similarity",
        "Min Similarity"

    ],

    "Value": [

        len(df),

        semantic_df.shape[1],

        semantic_encoded.shape[1],

        semantic_similarity.mean(),

        semantic_similarity.max(),

        semantic_similarity.min()

    ]

})

overlap_stats.to_csv(

    "outputs/tables/semantic_overlap_statistics.csv",

    index=False

)

print("\nOverlap statistics saved.")

print(overlap_stats)

# ============================================================
# RELATION CONTRIBUTION ANALYSIS
# ============================================================

relation_scores = pd.DataFrame({

    "Relation": [

        "Material",
        "Family",
        "Topology",
        "Complexity",
        "Structure"

    ],

    "Weight": [

        1,
        1,
        1,
        1,
        1

    ]

})

relation_scores.to_csv(

    "outputs/tables/semantic_relation_scores.csv",

    index=False

)

print("\nRelation scores saved.")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n====================================")
print("SEMANTIC SIMILARITY COMPLETE")
print("====================================")

print("\nGenerated Files:")

print("- outputs/tables/semantic_similarity_matrix.csv")
print("- outputs/tables/semantic_overlap_statistics.csv")
print("- outputs/tables/semantic_relation_scores.csv")