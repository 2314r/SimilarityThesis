# ============================================================
# ONTOLOGY SIMILARITY COMPUTATION
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs("outputs/tables", exist_ok=True)

print("Output folders ready.")

# ============================================================
# LOAD SEMANTIC SIMILARITY MATRIX
# ============================================================

similarity_path = "outputs/tables/semantic_similarity_matrix.csv"

similarity_df = pd.read_csv(

    similarity_path,

    index_col=0

)

print("\nSemantic similarity matrix loaded.")
print("Shape:", similarity_df.shape)

# ============================================================
# NORMALIZE SIMILARITY
# ============================================================

scaler = MinMaxScaler()

normalized_similarity = scaler.fit_transform(
    similarity_df
)

normalized_df = pd.DataFrame(

    normalized_similarity,

    index=similarity_df.index,

    columns=similarity_df.columns

)

# ============================================================
# SAVE NORMALIZED SIMILARITY
# ============================================================

normalized_df.to_csv(

    "outputs/tables/ontology_similarity_scores.csv"

)

print("\nNormalized ontology similarity saved.")

# ============================================================
# GENERATE ONTOLOGY RETRIEVAL RESULTS
# ============================================================

top_k = 5

retrieval_results = []

for assembly in normalized_df.index:

    similarities = normalized_df.loc[assembly]

    similarities = similarities.drop(assembly)

    top_neighbors = similarities.sort_values(

        ascending=False

    ).head(top_k)

    for neighbor, score in top_neighbors.items():

        retrieval_results.append({

            "QueryAssembly": assembly,
            "RetrievedAssembly": neighbor,
            "OntologySimilarity": score

        })

retrieval_df = pd.DataFrame(
    retrieval_results
)

retrieval_df.to_csv(

    "outputs/tables/ontology_retrieval_results.csv",

    index=False

)

print("\nOntology retrieval results saved.")

# ============================================================
# TOP NEIGHBORS TABLE
# ============================================================

top_neighbors_df = retrieval_df.groupby(

    "QueryAssembly"

).head(3)

top_neighbors_df.to_csv(

    "outputs/tables/ontology_top_neighbors.csv",

    index=False

)

print("\nTop ontology neighbors saved.")

# ============================================================
# SIMILARITY STATISTICS
# ============================================================

stats_df = pd.DataFrame({

    "Metric": [

        "Assemblies",
        "Mean Ontology Similarity",
        "Max Ontology Similarity",
        "Min Ontology Similarity",
        "Top-K Retrieval"

    ],

    "Value": [

        len(normalized_df),

        normalized_df.values.mean(),

        normalized_df.values.max(),

        normalized_df.values.min(),

        top_k

    ]

})

stats_df.to_csv(

    "outputs/tables/ontology_similarity_statistics.csv",

    index=False

)

print("\nOntology similarity statistics saved.")

print(stats_df)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n====================================")
print("ONTOLOGY SIMILARITY COMPLETE")
print("====================================")

print("\nGenerated Files:")

print("- outputs/tables/ontology_similarity_scores.csv")
print("- outputs/tables/ontology_retrieval_results.csv")
print("- outputs/tables/ontology_top_neighbors.csv")
print("- outputs/tables/ontology_similarity_statistics.csv")