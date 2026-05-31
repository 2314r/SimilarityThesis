"""
Run this script locally (not in sandbox) — it reads your assembly JSON files
from ~/Desktop/fusion_subset/ and extracts any real metadata for independent labels.

Usage:
    cd ~/Desktop/SimilarityThesis
    python3 extract_real_labels.py
"""

import os, json, pandas as pd
from pathlib import Path

DATASET_CSV  = Path("ontology/data/fusion360_rotor_dataset.csv")
FUSION_DIR   = Path.home() / "Desktop" / "fusion_subset"
OUTPUT_CSV   = Path("ontology/data/fusion360_with_real_labels.csv")

df = pd.read_csv(DATASET_CSV)
print(f"Loaded {len(df)} assemblies")

# Walk one sample to see what fields exist
sample_id   = df['assembly_name'].iloc[0]
sample_path = FUSION_DIR / sample_id / "assembly.json"

print(f"\nReading sample: {sample_path}")
with open(sample_path) as f:
    sample = json.load(f)

print("\nTop-level keys in assembly.json:")
for k, v in sample.items():
    if not isinstance(v, (dict, list)):
        print(f"  {k!r}: {v!r}")
    elif isinstance(v, list):
        print(f"  {k!r}: list of {len(v)}")
    else:
        print(f"  {k!r}: dict with keys {list(v.keys())[:5]}")

# Now extract metadata for ALL assemblies
rows = []
missing = []
for assembly_id in df['assembly_name']:
    json_path = FUSION_DIR / assembly_id / "assembly.json"
    if not json_path.exists():
        missing.append(assembly_id)
        rows.append({'assembly_name': assembly_id, 'json_found': False})
        continue

    with open(json_path) as f:
        data = json.load(f)

    row = {'assembly_name': assembly_id, 'json_found': True}
    # Extract any useful metadata fields — adjust based on actual keys found
    for key in ['description','type','category','industry','tags','label',
                'name','title','function','product_type','design_intent']:
        if key in data:
            row[key] = data[key]
    # Also check nested 'properties' or 'metadata' dicts
    for nested in ['properties','metadata','info','header']:
        if nested in data and isinstance(data[nested], dict):
            for k, v in data[nested].items():
                if not isinstance(v, (dict, list)):
                    row[f"{nested}_{k}"] = v
    rows.append(row)

meta_df = pd.DataFrame(rows)

# Merge with original
result = df.merge(meta_df, on='assembly_name', how='left')
result.to_csv(OUTPUT_CSV, index=False)

print(f"\n{'='*60}")
print(f"Total assemblies:   {len(df)}")
print(f"JSON files found:   {meta_df['json_found'].sum()}")
print(f"JSON files missing: {len(missing)}")
print(f"\nMetadata columns extracted: {[c for c in meta_df.columns if c not in ['assembly_name','json_found']]}")
print(f"\nSaved to: {OUTPUT_CSV}")

# Show unique values for any category-like columns
for col in meta_df.columns:
    if col in ['assembly_name','json_found']: continue
    n_unique = meta_df[col].nunique()
    print(f"\n  {col!r} — {n_unique} unique values:")
    print("   ", meta_df[col].value_counts().head(10).to_dict())
