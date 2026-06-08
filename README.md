# Machine Learning Algorithms for Assessing the Similarity of CAD Assembly Variants Based on Ontological Features

## Overview

This project implements a hybrid similarity assessment framework for CAD assemblies, combining technical, semantic, and structural knowledge within an ontology-based pipeline. The framework is evaluated on 746 Fusion 360 assemblies extracted from the Autodesk open-source repository.

---

## Research Objectives

- RQ1: How can CAD assemblies be represented using ontology-based features?
- RQ2: Which similarity metrics are most suitable for assembly retrieval?
- RQ3: Does a hybrid similarity approach outperform single-domain methods?
- RQ4: How can ontological knowledge support explainable engineering decisions?
- RQ5: How can similarity assessment support engineering decision-making?

---

## Methodology

### Feature Representation

| Feature Type | Description | Features |
|---|---|---|
| Technical | Geometric properties of the assembly | volume, edge_count, face_count, vertex_count, complexity_ratio, geometric_complexity (6 features; `surface_area` and `mass` removed — Pitfall B2: surface_area was zero-variance, mass ≈ volume with r = 0.998) |
| Semantic | Ontology-derived categorical knowledge (one-hot encoded) | material_class (Steel, Aluminum, Plastic, Wood, Other), topology_class, complexity_class, structure_class (4 features) |
| Structural | Assembly contact topology (used only for ablation; defines the labels, so excluded from the scored hybrid) | n_contacts, n_holes, n_occurrences |

### Hybrid Similarity Model

```
S_hybrid = α · S_technical + β · S_semantic + γ · S_structural
```

Weights selected by **5-fold cross-validation** (tuned on train folds, scored on held-out queries): **α = 0.90, β = 0.10, γ = 0.00** (chosen in 4/5 folds). Held-out MRR = 0.5726 vs in-sample 0.5731 → optimism bias = 0.0005.

### Ground-Truth Labels

Five contact-topology families derived from assembly JSON files (independent of geometric features):

| Family | Count |
|---|---|
| topo_isolated | 128 |
| topo_minimal | 134 |
| topo_moderate | 174 |
| topo_connected | 154 |
| topo_complex | 156 |

Independence verified: Decision Tree accuracy on geometric features = 39.4% vs. 20.0% chance level (5 balanced families).

---

## Results Summary

### Retrieval Performance

| Method | P@5 | MRR |
|---|---|---|
| **Hybrid (α=0.90, β=0.10, γ=0.00)** | **0.3542** | **0.5731** |
| Technical only | 0.3617 | 0.5640 |
| Semantic (ontology) | 0.3070 | 0.5405 |
| Euclidean baseline | 0.3362 | 0.5349 |
| Structural | 0.2735 | 0.4858 |
| Jaccard (raw material) | 0.2102 | 0.4244 |
| Random baseline | 0.1928 | 0.3920 |

In-sample Hybrid MRR = 0.5731; cross-validated held-out MRR = **0.5726** (95% bootstrap CI **[0.5478, 0.5999]**), optimism bias = 0.0005.

All pairwise comparisons (Wilcoxon signed-rank): **p < 0.05**

### Key Findings

- Hybrid similarity achieves highest MRR (0.5731), outperforming all single-domain methods
- Statistical significance confirmed for all comparisons (p < 0.05)
- Robustness: Hybrid MRR degrades from 0.5731 to 0.4713 under 50% semantic noise, remaining above Semantic-only throughout
- Bootstrap CI [0.5462–0.5999] lies entirely above the Semantic-only upper bound

### Ontology

- 746 assemblies encoded as OWL individuals
- 4,476 RDF triples
- Class hierarchy: TopoIsolated / TopoMinimal / TopoModerate / TopoConnected / TopoComplex → EngineeringAssembly
- SPARQL-queryable knowledge graph for explainable retrieval

---

## Technologies

- Python, NumPy, Pandas, Scikit-learn, Matplotlib
- rdflib (OWL/RDF ontology)
- scipy (statistical tests)

---

## How to Run

1. Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib rdflib scipy
```

2. Open and run the main notebook:
```bash
jupyter notebook ThesisV10.ipynb
```

3. Execute all cells sequentially. Outputs are saved to `ontology/outputs/`.

---

## Project Structure

```
SimilarityThesis/
├── ThesisV10.ipynb                    # Main canonical notebook
├── ontology/
│   ├── data/
│   │   └── fusion360_independent_labels.csv   # Dataset (746 assemblies, 19 features)
│   └── outputs/
│       ├── figures/                   # 15 publication-quality figures (PDF)
│       ├── tables/                    # 34 result CSVs
│       └── assembly_ontology.owl      # OWL ontology (4,476 triples)
└── archive/                           # Superseded notebooks and datasets
```

---

## Key Contribution

> A hybrid ontology-based similarity framework, evaluated under leakage-free conditions with cross-validated (held-out) weight selection, achieves statistically significant improvement in CAD assembly retrieval (in-sample MRR = 0.5731; held-out MRR = 0.5726, 95% CI [0.5478, 0.5999]; optimism bias = 0.0005) over all single-domain baselines, while an OWL/SPARQL knowledge graph provides explainable, auditable retrieval. The work also documents and corrects three evaluation-circularity pitfalls (label leakage, zero-variance features, in-sample weight optimization).

---

## Author

Ulukbek Rahmanov  
BSc Software Engineering  
University of Europe for Applied Sciences
