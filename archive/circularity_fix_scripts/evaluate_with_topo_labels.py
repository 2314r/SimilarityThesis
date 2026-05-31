"""
Re-evaluate all similarity methods using the independent topology-based labels.
Run locally:
    cd ~/Desktop/SimilarityThesis
    python3 evaluate_with_topo_labels.py
"""
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

DATA_PATH = "ontology/data/fusion360_independent_labels.csv"
df = pd.read_csv(DATA_PATH)

ASSEMBLY_NAMES = df['assembly_name'].tolist()
FAMILY_LOOKUP  = dict(zip(df['assembly_name'], df['topo_family']))  # ← NEW LABELS
N = len(ASSEMBLY_NAMES)

# ── Features (identical to ThesisV10) ────────────────────────────────────────
TECHNICAL_COLS  = ['volume','edge_count','face_count','vertex_count',
                   'complexity_ratio','geometric_complexity']
STRUCTURAL_COLS = ['component_count','body_count','rotational_symmetry']
SEMANTIC_COLS   = ['material_class','topology_class','complexity_class','structure_class']

df['topology_class']   = df['edge_count'].apply(
    lambda x: 'Sparse' if x<300 else ('Moderate' if x<1500 else 'Dense'))
df['complexity_class'] = df['geometric_complexity'].apply(
    lambda x: 'Low' if x<500 else ('Medium' if x<3000 else 'High'))
df['structure_class']  = np.where(df['rotational_symmetry']==1,'Symmetric','Asymmetric')
def nm(m):
    m=str(m).lower()
    for k,v in [('steel','Steel'),('aluminum','Aluminum'),('plastic','Plastic'),
                ('wood','Wood'),('composite','Composite')]:
        if k in m: return v
    return 'Other'
df['material_class'] = df['material'].apply(nm)

scaler = MinMaxScaler()
X_tech   = scaler.fit_transform(df[TECHNICAL_COLS].fillna(0))
X_struct = scaler.fit_transform(df[STRUCTURAL_COLS].fillna(0))
X_sem    = pd.get_dummies(df[SEMANTIC_COLS]).values.astype(float)

SIM_TECH   = cosine_similarity(X_tech);   np.fill_diagonal(SIM_TECH,1)
SIM_STRUCT = cosine_similarity(X_struct); np.fill_diagonal(SIM_STRUCT,1)
SIM_SEM    = cosine_similarity(X_sem);    np.fill_diagonal(SIM_SEM,1)
SIM_HYBRID = 0.0*SIM_TECH + 0.2*SIM_SEM + 0.8*SIM_STRUCT; np.fill_diagonal(SIM_HYBRID,1)

D_euc  = euclidean_distances(np.concatenate([X_tech,X_struct],axis=1))
SIM_EUC = 1 - D_euc/D_euc.max(); np.fill_diagonal(SIM_EUC,1)

np.random.seed(42)
_r = np.random.rand(N,N); SIM_RAND=(_r+_r.T)/2; np.fill_diagonal(SIM_RAND,1)

_X_mat = pd.get_dummies(df[['material']]).values.astype(float)
SIM_JAC = cosine_similarity(_X_mat); np.fill_diagonal(SIM_JAC,1)

# ── Metrics ───────────────────────────────────────────────────────────────────
def p_at_k(mat, k):
    s=[]
    for i,nm2 in enumerate(ASSEMBLY_NAMES):
        row=mat[i].copy(); row[i]=-1
        top=np.argsort(row)[::-1][:k]
        s.append(sum(1 for j in top if FAMILY_LOOKUP.get(ASSEMBLY_NAMES[j])==FAMILY_LOOKUP.get(nm2))/k)
    return np.mean(s)

def mrr(mat):
    s=[]
    for i,nm2 in enumerate(ASSEMBLY_NAMES):
        row=mat[i].copy(); row[i]=-1
        for rank,j in enumerate(np.argsort(row)[::-1],1):
            if FAMILY_LOOKUP.get(ASSEMBLY_NAMES[j])==FAMILY_LOOKUP.get(nm2):
                s.append(1/rank); break
        else: s.append(0)
    return np.mean(s)

METHODS = {
    'Hybrid':              SIM_HYBRID,
    'Structural':          SIM_STRUCT,
    'Semantic (Ontology)': SIM_SEM,
    'Technical':           SIM_TECH,
    'Euclidean':           SIM_EUC,
    'Jaccard (raw)':       SIM_JAC,
    'Random':              SIM_RAND,
}

print(f"{'Method':<24} {'P@3':>6} {'P@5':>6} {'P@10':>6} {'MRR':>7}")
print("─"*52)
results = {}
for name, mat in METHODS.items():
    p3=p_at_k(mat,3); p5=p_at_k(mat,5); p10=p_at_k(mat,10); m=mrr(mat)
    results[name] = {'P@3':p3,'P@5':p5,'P@10':p10,'MRR':m}
    print(f"{name:<24} {p3:>6.4f} {p5:>6.4f} {p10:>6.4f} {m:>7.4f}")

# Hybrid vs others Wilcoxon
print("\nHybrid vs baselines (Wilcoxon, MRR per-query):")
def mrr_vec(mat):
    s=[]
    for i,nm2 in enumerate(ASSEMBLY_NAMES):
        row=mat[i].copy(); row[i]=-1
        for rank,j in enumerate(np.argsort(row)[::-1],1):
            if FAMILY_LOOKUP.get(ASSEMBLY_NAMES[j])==FAMILY_LOOKUP.get(nm2):
                s.append(1/rank); break
        else: s.append(0)
    return np.array(s)

h_mrr = mrr_vec(SIM_HYBRID)
for name, mat in METHODS.items():
    if name == 'Hybrid': continue
    o_mrr = mrr_vec(mat)
    _, pval = stats.wilcoxon(h_mrr, o_mrr, alternative='greater')
    delta = (h_mrr - o_mrr).mean()
    sig = '***' if pval<0.001 else ('**' if pval<0.01 else ('*' if pval<0.05 else 'n.s.'))
    print(f"  vs {name:<24} Δ={delta:+.4f}  p={pval:.2e}  {sig}")

# Random baseline (expected = largest class fraction)
largest = df['topo_family'].value_counts().iloc[0] / N
print(f"\nRandom baseline (expected ≈ largest class fraction): {largest:.4f}")
print(f"Random MRR achieved: {results['Random']['MRR']:.4f}")


# ── Re-optimize weights on independent labels ─────────────────────────────────
print("\n" + "="*60)
print("Grid search on INDEPENDENT labels (α+β+γ=1, step 0.1)")
print("="*60)

steps = np.arange(0.0, 1.01, 0.1)
best_mrr, best_a, best_b, best_g = -1, 0, 0, 1
grid_results = []

for a in steps:
    for b in steps:
        g = round(1.0 - a - b, 10)
        if g < -0.001 or g > 1.001: continue
        g = max(0.0, min(1.0, g))
        hybrid = a*SIM_TECH + b*SIM_SEM + g*SIM_STRUCT
        np.fill_diagonal(hybrid, 1)
        m = mrr(hybrid)
        grid_results.append({'alpha':round(a,2),'beta':round(b,2),'gamma':round(g,2),'MRR':round(m,4)})
        if m > best_mrr:
            best_mrr, best_a, best_b, best_g = m, a, b, g

grid_df = pd.DataFrame(grid_results).sort_values('MRR', ascending=False)
print(f"\nBest weights: α={best_a:.2f} (Tech)  β={best_b:.2f} (Sem)  γ={best_g:.2f} (Struct)")
print(f"Best hybrid MRR: {best_mrr:.4f}")
print(f"\nTop 10 weight combinations:")
print(grid_df.head(10).to_string(index=False))

# Compare re-tuned hybrid vs all single methods
SIM_HYBRID_NEW = best_a*SIM_TECH + best_b*SIM_SEM + best_g*SIM_STRUCT
np.fill_diagonal(SIM_HYBRID_NEW, 1)

print(f"\n{'Method':<28} {'MRR':>7}")
print("─"*36)
print(f"{'Hybrid (old weights α=0.0,β=0.2,γ=0.8)':<28} {results['Hybrid']['MRR']:>7.4f}")
print(f"{'Hybrid (new weights)':<28} {mrr(SIM_HYBRID_NEW):>7.4f}")
print(f"{'Technical (single)':<28} {results['Technical']['MRR']:>7.4f}")
print(f"{'Structural (single)':<28} {results['Structural']['MRR']:>7.4f}")
print(f"{'Semantic (single)':<28} {results['Semantic (Ontology)']['MRR']:>7.4f}")

h_mrr_new = mrr_vec(SIM_HYBRID_NEW)
for name, mat in [('Technical',SIM_TECH),('Structural',SIM_STRUCT),('Semantic (Ontology)',SIM_SEM)]:
    o_mrr = mrr_vec(mat)
    _, pval = stats.wilcoxon(h_mrr_new, o_mrr, alternative='greater')
    delta = (h_mrr_new - o_mrr).mean()
    sig = '***' if pval<0.001 else ('**' if pval<0.01 else ('*' if pval<0.05 else 'n.s.'))
    print(f"  New hybrid vs {name:<20} Δ={delta:+.4f}  p={pval:.2e}  {sig}")
