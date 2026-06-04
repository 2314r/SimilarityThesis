# Machine Learning Algorithms for Assessing the Similarity of Variant Rotor Designs Based on Ontological Features

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
| Technical | Geometric properties of the assembly | component_count, body_count, edge_count, face_count, vertex_count, mass, volume, geometric_complexity, complexity_ratio, rotational_symmetry |
| Semantic | Ontology-encoded material knowledge | material one-hot encoding (Steel, Aluminum, Plastic, Brass) |
| Structural | Assembly contact topology | n_contacts, n_holes, n_occurrences |

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
| **Hybrid (α=0.90, β=0.10, γ=0.00)** | **0.354** | **0.5731** |
| Technical only | 0.362 | 0.5640 |
| Semantic (ontology) | 0.307 | 0.5410 |
| Structural | 0.274 | 0.4860 |
| Euclidean baseline | 0.336 | 0.5350 |
| Random baseline | — | 0.3920 |

Bootstrap 95% CI for Hybrid MRR: **[0.5462, 0.5999]**

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
│       ├── figures/                   # 28 publication-quality figures
│       ├── tables/                    # 31 result CSVs
│       └── assembly_ontology.owl      # OWL ontology (4,476 triples)
└── archive/                           # Superseded notebooks and datasets
```

---

## Key Contribution

> A hybrid ontology-based similarity framework combining geometric, semantic, and structural knowledge achieves statistically significant improvement in CAD assembly retrieval (MRR = 0.5731, 95% CI [0.5462, 0.5999]) over all single-domain baselines.

---

## Author

Ulukbek Rahmanov  
BSc Software Engineering  
University of Europe for Applied Sciences
