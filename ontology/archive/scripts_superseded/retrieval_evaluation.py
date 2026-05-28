# ============================================================
# STEP 2D — ONTOLOGY-AWARE RETRIEVAL EVALUATION
# Ontology-Guided Hybrid CAD Assembly Retrieval Framework
# Target: Advanced Engineering Informatics (Elsevier)
#
# Compares: Technical vs Ontology vs Hybrid retrieval
# Metrics: Precision@3, Precision@5, Precision@10, MRR
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
print("Output folders ready.\n")

# ============================================================
# LOAD DATASET — Ground Truth Labels (family)
# ============================================================

df = pd.read_csv("fusion360_rotor_dataset.csv")
# Build family lookup: assembly_name -> family
family_lookup = dict(zip(df['assembly_name'], df['family']))
assembly_names = sorted(df['assembly_name'].tolist())
print(f"Dataset: {len(assembly_names)} assemblies")
print(f"Families: {df['family'].value_counts().to_dict()}\n")

# ============================================================
# LOAD SIMILARITY MATRICES
# ============================================================

print("Loading similarity matrices...")
tech_df  = pd.read_csv("outputs/tables/technical_similarity_matrix.csv",  index_col=0)
onto_df  = pd.read_csv("outputs/tables/ontology_similarity_scores.csv",   index_col=0)
hybrid_df = pd.read_csv("outputs/tables/hybrid_similarity_matrix.csv",    index_col=0)

# Align to common assemblies
common = sorted(set(tech_df.index) & set(onto_df.index) & set(hybrid_df.index))
print(f"Common assemblies across all matrices: {len(common)}\n")

tech_arr   = tech_df.loc[common, common].values.astype(float)
onto_arr   = onto_df.loc[common, common].values.astype(float)
hybrid_arr = hybrid_df.loc[common, common].values.astype(float)

# ============================================================
# RETRIEVAL EVALUATION FUNCTIONS
# ============================================================

def precision_at_k(sim_matrix, labels, assemblies, k):
    """
    Precision@K: fraction of top-K retrieved items that share
    the same engineering family as the query.
    """
    precisions = []
    for i, name in enumerate(assemblies):
        query_family = labels.get(name)
        if query_family is None:
            continue
        row = sim_matrix[i].copy()
        row[i] = -1  # exclude self
        top_k_idx = np.argsort(row)[::-1][:k]
        retrieved_families = [labels.get(assemblies[j]) for j in top_k_idx]
        hits = sum(1 for f in retrieved_families if f == query_family)
        precisions.append(hits / k)
    return float(np.mean(precisions))


def mean_reciprocal_rank(sim_matrix, labels, assemblies):
    """
    MRR: mean of 1/rank of the first correctly retrieved item.
    """
    rr_list = []
    for i, name in enumerate(assemblies):
        query_family = labels.get(name)
        if query_family is None:
            continue
        row = sim_matrix[i].copy()
        row[i] = -1
        ranked_idx = np.argsort(row)[::-1]
        rr = 0.0
        for rank, j in enumerate(ranked_idx, start=1):
            if labels.get(assemblies[j]) == query_family:
                rr = 1.0 / rank
                break
        rr_list.append(rr)
    return float(np.mean(rr_list))

# ============================================================
# COMPUTE METRICS FOR ALL THREE METHODS
# ============================================================

methods = {
    'Technical':  tech_arr,
    'Ontology':   onto_arr,
    'Hybrid':     hybrid_arr,
}

results = []
for method_name, mat in methods.items():
    p3  = precision_at_k(mat, family_lookup, common, k=3)
    p5  = precision_at_k(mat, family_lookup, common, k=5)
    p10 = precision_at_k(mat, family_lookup, common, k=10)
    mrr = mean_reciprocal_rank(mat, family_lookup, common)
    results.append({
        'Method':       method_name,
        'Precision@3':  round(p3,  4),
        'Precision@5':  round(p5,  4),
        'Precision@10': round(p10, 4),
        'MRR':          round(mrr, 4),
    })
    print(f"{method_name:12s}  P@3={p3:.4f}  P@5={p5:.4f}  P@10={p10:.4f}  MRR={mrr:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("outputs/tables/retrieval_comparison.csv", index=False)
print(f"\nSaved: retrieval_comparison.csv")

# ============================================================
# FIGURE 1 — BAR CHART: Precision@K Comparison
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Retrieval Performance Comparison\n'
             'Ontology-Guided Hybrid CAD Assembly Retrieval',
             fontsize=13, fontweight='bold', y=1.01)

colors = {'Technical': '#4C72B0', 'Ontology': '#DD8452', 'Hybrid': '#55A868'}
method_names = [r['Method'] for r in results]
bar_colors   = [colors[m] for m in method_names]

# Precision@K grouped bar
ks    = [3, 5, 10]
x     = np.arange(len(ks))
width = 0.25

ax = axes[0]
for i, r in enumerate(results):
    vals = [r['Precision@3'], r['Precision@5'], r['Precision@10']]
    ax.bar(x + i*width, vals, width, label=r['Method'],
           color=list(colors.values())[i], alpha=0.85, edgecolor='white')

ax.set_xticks(x + width)
ax.set_xticklabels(['Precision@3', 'Precision@5', 'Precision@10'])
ax.set_ylabel('Precision')
ax.set_ylim(0, 1.05)
ax.set_title('Precision@K by Method')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.4)

# MRR bar chart
ax2 = axes[1]
mrr_vals = [r['MRR'] for r in results]
bars = ax2.bar(method_names, mrr_vals, color=bar_colors, alpha=0.85,
               edgecolor='white', width=0.5)
for bar, val in zip(bars, mrr_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_ylabel('Mean Reciprocal Rank (MRR)')
ax2.set_ylim(0, 1.05)
ax2.set_title('MRR by Method')
ax2.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("outputs/figures/retrieval_comparison.pdf", bbox_inches='tight', dpi=150)
plt.savefig("outputs/figures/retrieval_comparison.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: retrieval_comparison.pdf / .png")

# ============================================================
# FIGURE 2 — HYBRID SIMILARITY MATRIX HEATMAP (sample 50)
# ============================================================

sample_n = 50
sample_idx = list(range(0, len(common), len(common)//sample_n))[:sample_n]
sample_names = [common[i] for i in sample_idx]
sample_hybrid = hybrid_arr[np.ix_(sample_idx, sample_idx)]

fig2, ax3 = plt.subplots(figsize=(10, 8))
im = ax3.imshow(sample_hybrid, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax3, label='Hybrid Similarity Score')
ax3.set_title('Hybrid Similarity Matrix (Sample of 50 Assemblies)\n'
              'Ontology-Guided CAD Assembly Retrieval', fontsize=12, fontweight='bold')
ax3.set_xlabel('Assembly Index')
ax3.set_ylabel('Assembly Index')
ax3.set_xticks([])
ax3.set_yticks([])
plt.tight_layout()
plt.savefig("outputs/figures/hybrid_similarity_heatmap.pdf", bbox_inches='tight', dpi=150)
plt.savefig("outputs/figures/hybrid_similarity_heatmap.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved: hybrid_similarity_heatmap.pdf / .png")

# ============================================================
# IMPROVEMENT TABLE — Hybrid vs Individual
# ============================================================

tech_row    = next(r for r in results if r['Method'] == 'Technical')
onto_row    = next(r for r in results if r['Method'] == 'Ontology')
hybrid_row  = next(r for r in results if r['Method'] == 'Hybrid')

improvement = pd.DataFrame([
    {
        'Comparison': 'Hybrid vs Technical',
        'Delta_P@3':  round(hybrid_row['Precision@3']  - tech_row['Precision@3'],  4),
        'Delta_P@5':  round(hybrid_row['Precision@5']  - tech_row['Precision@5'],  4),
        'Delta_P@10': round(hybrid_row['Precision@10'] - tech_row['Precision@10'], 4),
        'Delta_MRR':  round(hybrid_row['MRR']          - tech_row['MRR'],          4),
    },
    {
        'Comparison': 'Hybrid vs Ontology',
        'Delta_P@3':  round(hybrid_row['Precision@3']  - onto_row['Precision@3'],  4),
        'Delta_P@5':  round(hybrid_row['Precision@5']  - onto_row['Precision@5'],  4),
        'Delta_P@10': round(hybrid_row['Precision@10'] - onto_row['Precision@10'], 4),
        'Delta_MRR':  round(hybrid_row['MRR']          - onto_row['MRR'],          4),
    },
])
improvement.to_csv("outputs/tables/retrieval_improvement.csv", index=False)
print("Saved: retrieval_improvement.csv")

print("\n--- Retrieval Improvement ---")
print(improvement.to_string(index=False))

# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print("\n" + "="*60)
print("STEP 2D COMPLETE — RETRIEVAL EVALUATION RESULTS")
print("="*60)
print(results_df.to_string(index=False))
print("="*60)
print("\n✅ All outputs saved to outputs/tables/ and outputs/figures/")
print("\n📌 IMPORTANT for AEI paper positioning:")
print("   The hybrid system's contribution is EXPLAINABILITY")
print("   and SEMANTIC ENGINEERING REPRESENTATION,")
print("   not raw accuracy improvement.")
