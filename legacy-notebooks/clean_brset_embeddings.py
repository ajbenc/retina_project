"""
BRSET Embedding Cleaning Script
================================
Removes bad-quality images and embedding-based consensus outliers
from all 7 BRSET embedding CSVs.

Criteria:
1. Quality filter: quality == "Inadequate" (1,987 images)
2. Embedding consensus: IQR outliers on L2 norms flagged by >= 3 of 7 embeddings (25 images)
Total removed: ~2,012 images (12.4%)

Output: data/brset_embeddings_cleaned/ with same file structure
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import glob
import shutil

PROJECT_ROOT = Path.cwd()
SRC_DIR = PROJECT_ROOT / "data" / "brset_embeddings"
DST_DIR = PROJECT_ROOT / "data" / "brset_embeddings_cleaned"
DST_DIR.mkdir(parents=True, exist_ok=True)

# === Step 1: Load labels and find quality-bad images ===
labels = pd.read_csv(SRC_DIR / "brset_labels" / "labels_brset.csv")
print(f"Original: {len(labels)} images, {labels['patient_id'].nunique()} patients")

bad_quality_ids = set(labels[labels['quality'] == 'Inadequate']['image_id'].astype(str))
print(f"Inadequate quality: {len(bad_quality_ids)} images")

# === Step 2: Embedding-based outlier detection (on quality-clean only) ===
clean_ids = set(labels[labels['quality'] == 'Adequate']['image_id'].astype(str))
emb_files = sorted(SRC_DIR.glob("Embeddings_*.csv"))

all_outlier_ids = {}
for ef in emb_files:
    emb = pd.read_csv(ef)
    
    if 'name' in emb.columns:
        id_col = 'name'
        feat_cols = [c for c in emb.columns if c.startswith('feature_')]
    elif 'ImageName' in emb.columns:
        id_col = 'ImageName'
        feat_cols = [c for c in emb.columns if c not in ['ImageName', 'Unnamed: 0']]
    else:
        continue
    
    emb['_id'] = emb[id_col].astype(str).str.replace('.jpg', '', regex=False).str.replace('.png', '', regex=False)
    emb_clean = emb[emb['_id'].isin(clean_ids)]
    
    feats = emb_clean[feat_cols].values.astype(np.float32)
    norms = np.linalg.norm(feats, axis=1)
    
    q1, q3 = np.percentile(norms, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    outlier_mask = (norms < lower) | (norms > upper)
    outlier_ids = set(emb_clean.loc[outlier_mask, '_id'].values)
    
    short = ef.stem.replace('Embeddings_brset_', '')
    all_outlier_ids[short] = outlier_ids
    print(f"  {short:30s}: {len(outlier_ids)} norm outliers")

# Consensus >= 3
all_ids_flat = []
for ids in all_outlier_ids.values():
    all_ids_flat.extend(ids)
id_counts = Counter(all_ids_flat)
consensus_outliers = {k for k, v in id_counts.items() if v >= 3}
print(f"Consensus outliers (>=3 embeddings): {len(consensus_outliers)}")

# === Step 3: Combine and remove ===
final_remove = bad_quality_ids | consensus_outliers
print(f"Total images to remove: {len(final_remove)}")

# Clean labels
clean_labels = labels[~labels['image_id'].astype(str).isin(final_remove)].copy()
print(f"Clean labels: {len(clean_labels)} images, {clean_labels['patient_id'].nunique()} patients")
print(f"Glaucoma+: {int(clean_labels['increased_cup_disc'].sum())} ({clean_labels['increased_cup_disc'].mean():.1%})")

# Save cleaned labels
labels_dst = DST_DIR / "brset_labels"
labels_dst.mkdir(parents=True, exist_ok=True)
clean_labels.to_csv(labels_dst / "labels_brset.csv", index=False)
print(f"Saved: {labels_dst / 'labels_brset.csv'}")

# === Step 4: Clean each embedding CSV ===
for ef in emb_files:
    emb = pd.read_csv(ef)
    
    if 'name' in emb.columns:
        id_col = 'name'
    elif 'ImageName' in emb.columns:
        id_col = 'ImageName'
    else:
        continue
    
    emb['_id'] = emb[id_col].astype(str).str.replace('.jpg', '', regex=False).str.replace('.png', '', regex=False)
    emb_clean = emb[~emb['_id'].isin(final_remove)].drop(columns=['_id'])
    
    dst_path = DST_DIR / ef.name
    emb_clean.to_csv(dst_path, index=False)
    print(f"  {ef.name}: {len(emb)} -> {len(emb_clean)} ({len(emb)-len(emb_clean)} removed) -> {dst_path.name}")

# === Step 5: Save cleaning summary ===
summary = pd.DataFrame({
    'criterion': ['Inadequate quality', 'Embedding consensus (>=3)', 'Total removed', 'Remaining images', 'Remaining patients', 'Glaucoma+ remaining'],
    'count': [len(bad_quality_ids), len(consensus_outliers), len(final_remove), len(clean_labels), clean_labels['patient_id'].nunique(), int(clean_labels['increased_cup_disc'].sum())],
})
summary.to_csv(DST_DIR / "cleaning_summary.csv", index=False)
print(f"\nSaved cleaning summary: {DST_DIR / 'cleaning_summary.csv'}")
print("\nDone!")
