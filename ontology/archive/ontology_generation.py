# ============================================================
# ONTOLOGY-GUIDED ENGINEERING KNOWLEDGE GENERATION
# ============================================================

import os
import pandas as pd
import numpy as np

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)

print("Output folders ready.")

# ============================================================
# LOAD DATASET
# ============================================================

DATASET_PATH = "fusion360_rotor_dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("\nDataset Loaded Successfully.")
print("Dataset Shape:", df.shape)

# ============================================================
# COPY DATASET
# ============================================================

ontology_df = df.copy()

# ============================================================
# COMPLEXITY CLASSIFICATION
# ============================================================

def complexity_class(x):

    if x < 500:
        return "Low"

    elif x < 3000:
        return "Medium"

    return "High"

ontology_df["complexity_level"] = ontology_df[
    "geometric_complexity"
].apply(complexity_class)

# ============================================================
# TOPOLOGY CLASSIFICATION
# ============================================================

def topology_class(x):

    if x < 300:
        return "Sparse"

    elif x < 1500:
        return "Moderate"

    return "Dense"

ontology_df["topology_level"] = ontology_df[
    "edge_count"
].apply(topology_class)

# ============================================================
# GENERATE ONTOLOGY TRIPLES
# ============================================================

triples = []

for idx, row in ontology_df.iterrows():

    assembly = row["assembly_name"]

    # --------------------------------------------------------
    # MATERIAL RELATION
    # --------------------------------------------------------

    triples.append([

        assembly,

        "hasMaterial",

        row["material"]

    ])

    # --------------------------------------------------------
    # FAMILY RELATION
    # --------------------------------------------------------

    triples.append([

        assembly,

        "belongsToFamily",

        row["family"]

    ])

    # --------------------------------------------------------
    # COMPLEXITY RELATION
    # --------------------------------------------------------

    triples.append([

        assembly,

        "hasComplexityLevel",

        row["complexity_level"]

    ])

    # --------------------------------------------------------
    # TOPOLOGY RELATION
    # --------------------------------------------------------

    triples.append([

        assembly,

        "hasTopology",

        row["topology_level"]

    ])

    # --------------------------------------------------------
    # STRUCTURAL RELATION
    # --------------------------------------------------------

    structure = "Symmetric"

    if row["rotational_symmetry"] == 0:
        structure = "Asymmetric"

    triples.append([

        assembly,

        "hasStructure",

        structure

    ])

# ============================================================
# CREATE TRIPLE DATAFRAME
# ============================================================

triples_df = pd.DataFrame(

    triples,

    columns=[

        "Subject",

        "Predicate",

        "Object"

    ]

)

# ============================================================
# SAVE ONTOLOGY TRIPLES
# ============================================================

triples_df.to_csv(

    "tables/ontology_triples.csv",

    index=False

)

print("\nOntology triples saved.")

print("\nTriples Preview:\n")

print(triples_df.head())

# ============================================================
# GENERATE ONTOLOGY NODES
# ============================================================

nodes = set()

for _, row in triples_df.iterrows():

    nodes.add(row["Subject"])
    nodes.add(row["Object"])

nodes_df = pd.DataFrame({

    "Node": list(nodes)

})

nodes_df.to_csv(

    "tables/ontology_nodes.csv",

    index=False

)

print("\nOntology nodes saved.")

# ============================================================
# GENERATE ONTOLOGY EDGES
# ============================================================

edges_df = triples_df.copy()

edges_df.to_csv(

    "tables/ontology_edges.csv",

    index=False

)

print("\nOntology edges saved.")

# ============================================================
# ONTOLOGY STATISTICS
# ============================================================

ontology_stats = pd.DataFrame({

    "Metric": [

        "Total Triples",
        "Unique Nodes",
        "Unique Predicates",
        "Assemblies",
        "Material Classes",
        "Family Classes"

    ],

    "Value": [

        len(triples_df),

        len(nodes_df),

        triples_df["Predicate"].nunique(),

        ontology_df["assembly_name"].nunique(),

        ontology_df["material"].nunique(),

        ontology_df["family"].nunique()

    ]

})

ontology_stats.to_csv(

    "tables/ontology_statistics.csv",

    index=False

)

print("\nOntology statistics saved.")

print("\nOntology Statistics:\n")

print(ontology_stats)

# ============================================================
# RELATION DISTRIBUTION
# ============================================================

relation_distribution = triples_df[
    "Predicate"
].value_counts().reset_index()

relation_distribution.columns = [

    "Relation",
    "Count"

]

relation_distribution.to_csv(

    "tables/semantic_relation_distribution.csv",

    index=False

)

print("\nSemantic relation distribution saved.")

print("\nRelation Distribution:\n")

print(relation_distribution)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n====================================")
print("ONTOLOGY GENERATION COMPLETED")
print("====================================")

print("\nGenerated Files:")

print("- tables/ontology_triples.csv")
print("- tables/ontology_nodes.csv")
print("- tables/ontology_edges.csv")
print("- tables/ontology_statistics.csv")
print("- tables/semantic_relation_distribution.csv")

print("\nOntology implementation successful.")