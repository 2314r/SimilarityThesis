# ============================================================
# ENGINEERING KNOWLEDGE GRAPH GENERATION
# ============================================================

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/graphs", exist_ok=True)

print("Output folders ready.")

# ============================================================
# LOAD ONTOLOGY TRIPLES
# ============================================================

triples_path = "outputs/tables/ontology_triples.csv"

triples_df = pd.read_csv(triples_path)

print("\nOntology triples loaded.")
print("Shape:", triples_df.shape)

# ============================================================
# BUILD KNOWLEDGE GRAPH
# ============================================================

G = nx.Graph()

for _, row in triples_df.iterrows():

    subject = row["Subject"]
    predicate = row["Predicate"]
    obj = row["Object"]

    # Add nodes
    G.add_node(subject)
    G.add_node(obj)

    # Add semantic edge
    G.add_edge(

        subject,

        obj,

        relation=predicate

    )

print("\nKnowledge graph created.")

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# ============================================================
# SAVE GRAPH STRUCTURE
# ============================================================

graph_path = "outputs/graphs/ontology_graph.graphml"

nx.write_graphml(

    G,

    graph_path

)

print("\nGraphML saved.")

# ============================================================
# GRAPH STATISTICS
# ============================================================

graph_stats = pd.DataFrame({

    "Metric": [

        "Total Nodes",
        "Total Edges",
        "Graph Density",
        "Connected Components",
        "Average Degree"

    ],

    "Value": [

        G.number_of_nodes(),

        G.number_of_edges(),

        nx.density(G),

        nx.number_connected_components(G),

        sum(dict(G.degree()).values()) / G.number_of_nodes()

    ]

})

graph_stats.to_csv(

    "outputs/tables/ontology_graph_statistics.csv",

    index=False

)

print("\nGraph statistics saved.")

print(graph_stats)

# ============================================================
# NODE CENTRALITY
# ============================================================

degree_centrality = nx.degree_centrality(G)

centrality_df = pd.DataFrame({

    "Node": list(degree_centrality.keys()),

    "Centrality": list(degree_centrality.values())

})

centrality_df = centrality_df.sort_values(

    by="Centrality",

    ascending=False

)

centrality_df.to_csv(

    "outputs/tables/ontology_node_centrality.csv",

    index=False

)

print("\nNode centrality saved.")

# ============================================================
# GRAPH VISUALIZATION
# ============================================================

# Use smaller subgraph for readable figure
sample_nodes = list(G.nodes())[:120]

subgraph = G.subgraph(sample_nodes)

plt.figure(figsize=(16,12))

pos = nx.spring_layout(

    subgraph,

    seed=42,

    k=0.6

)

nx.draw_networkx_nodes(

    subgraph,

    pos,

    node_size=120,

    alpha=0.8

)

nx.draw_networkx_edges(

    subgraph,

    pos,

    alpha=0.3

)

# Limited labels
labels_subset = {

    node: node

    for i, node in enumerate(subgraph.nodes())

    if i < 40

}

nx.draw_networkx_labels(

    subgraph,

    pos,

    labels_subset,

    font_size=7

)

plt.title(

    "Ontology-Guided Engineering Knowledge Graph"

)

plt.axis("off")

plt.tight_layout()

plt.savefig(

    "outputs/figures/ontology_graph_enhanced.pdf",

    bbox_inches="tight"

)

plt.close()

print("\nOntology graph figure saved.")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n====================================")
print("KNOWLEDGE GRAPH GENERATION COMPLETE")
print("====================================")

print("\nGenerated Files:")

print("- outputs/graphs/ontology_graph.graphml")
print("- outputs/tables/ontology_graph_statistics.csv")
print("- outputs/tables/ontology_node_centrality.csv")
print("- outputs/figures/ontology_graph_enhanced.pdf")