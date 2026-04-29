# Machine Learning Algorithms for Assessing the Similarity of Variant Rotor Designs Based on Ontological Features

## 📌 Overview

This project implements a hybrid similarity assessment framework for comparing rotor design variants. The approach integrates:

- Technical features (numerical properties such as diameter, speed, mass)
- Semantic features (ontology-inspired attributes such as function and components)
- Structural features (component relationships and topology)

The objective is to demonstrate that a hybrid similarity model outperforms single-domain similarity methods in engineering design comparison.

---

## 🎯 Research Objectives

The implementation addresses the following research questions:

- RQ1: How can rotor designs be represented using ontology-based features?
- RQ2: Which similarity metrics are most suitable?
- RQ3: Can machine learning models learn similarity relationships?
- RQ4: Does a hybrid similarity approach outperform single-domain methods?
- RQ5: How can similarity support engineering decision-making?

---

## 🧠 Methodology

### Feature Representation

Each rotor variant is represented using three feature categories:

| Feature Type  | Description | Example |
|--------------|------------|--------|
| Technical     | Numerical physical properties | diameter, speed |
| Semantic      | Functional / ontology features | motor, magnet |
| Structural    | Component relationships | number of components |

---

### Similarity Methods

The following similarity approaches are implemented:

- Cosine Similarity (technical features)
- Jaccard Similarity (semantic features)
- Structural Similarity (cosine on structural features)
- Hybrid Similarity Model:

[
S_{hybrid} = \alpha S_{semantic} + \beta S_{technical} + \gamma S_{structural}
]

---

### Evaluation Metrics

- Precision@K (K = 3, 5, 10)
- Mean Reciprocal Rank (MRR)
- ROC-AUC (for ML models)

---

## 📊 Results Summary

### Similarity Performance

| Method       | Precision@5 | Precision@10 | MRR |
|-------------|------------|--------------|-----|
| Technical    | 0.25       | 0.25         | 0.49 |
| Semantic     | 0.48       | 0.49         | 0.70 |
| Structural   | 0.51       | 0.47         | 0.75 |
| Hybrid   | 0.83   | 0.63     | 0.996 |

### Key Findings

- Hybrid similarity significantly outperforms individual methods
- High MRR indicates excellent ranking quality
- Semantic and structural features contribute strongly
- Technical features alone are insufficient

---

### Machine Learning Models

| Model         | ROC-AUC |
|--------------|--------|
| k-NN          | 0.96   |
| Random Forest | 1.00   |

👉 The feature representation is highly learnable and discriminative.

---

### Robustness (Missing Features)

| Condition         | Precision@5 |
|------------------|------------|
| Full Data         | 0.83       |
| Missing Features  | 0.81       |

👉 The hybrid model remains stable under incomplete data.

---

## 📈 Visualizations

The project includes:

- Precision@K comparison charts
- Similarity matrix heatmaps
- t-SNE embedding of rotor variants
- Ranking examples

---

## 🛠️ Technologies Used

- Python
- NumPy, Pandas
- Scikit-learn
- Matplotlib

---

## ▶️ How to Run

1. Install dependencies:
bash pip install numpy pandas scikit-learn matplotlib 

2. Run the notebook:
bash jupyter notebook 

3. Execute all cells sequentially.

---

## 📂 Project Structure

. ├── thesis.ipynb        # Main implementation ├── README.md           # Project documentation

---

## 🧠 Key Contribution

This project demonstrates that:

> Combining semantic, technical, and structural knowledge leads to more accurate and robust similarity assessment in engineering design.

---

## 📌 Author

Ulukbek Rahmanov  
BSc Software Engineering  
University of Europe for Applied Sciences

---

## 📎 Notes

- The ontology is implemented in a lightweight, feature-based form
- No external OWL tools are required
- All experiments are reproducible

---

## 🚀 Future Work

- Integration with real CAD datasets
- Full OWL/RDF ontology implementation
- Deep learning similarity models (Siamese networks)
- Graph-based similarity using real topology

--
