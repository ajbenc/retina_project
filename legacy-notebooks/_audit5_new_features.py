"""
Audit: optic_disc, macula, diabetes_time_y, insuline — for R27
"""
import pandas as pd
import numpy as np

labels = pd.read_csv('data/brset_embeddings_cleaned/brset_labels/labels_brset.csv')

print('='*80)
print('NEW FEATURES FOR R27')
print('='*80)

# 1. optic_disc
print('\n── optic_disc ──')
print(f'dtype: {labels["optic_disc"].dtype}')
print(f'Values: {labels["optic_disc"].value_counts().to_dict()}')
# Cross-tab with glaucoma
ct = pd.crosstab(labels['optic_disc'], labels['increased_cup_disc'], normalize='index')
print(f'\nGlaucoma rate by optic_disc:')
for val in sorted(labels['optic_disc'].unique()):
    sub = labels[labels['optic_disc'] == val]
    rate = sub['increased_cup_disc'].mean()
    print(f'  optic_disc={val!r}: {len(sub)} images, glaucoma rate = {rate:.1%}')

# ⚠️ LEAKAGE CHECK: is optic_disc CORRELATED with increased_cup_disc?
# If optic_disc is graded at the same time by the same grader, it could leak info
from sklearn.metrics import mutual_info_score
mi = mutual_info_score(labels['optic_disc'].astype(str), labels['increased_cup_disc'])
print(f'\n  Mutual info (optic_disc vs glaucoma): {mi:.4f}')
# If very high, it might be label leakage

# 2. macula
print('\n── macula ──')
print(f'dtype: {labels["macula"].dtype}')
print(f'Values: {labels["macula"].value_counts().to_dict()}')
ct2 = pd.crosstab(labels['macula'], labels['increased_cup_disc'], normalize='index')
for val in sorted(labels['macula'].unique()):
    sub = labels[labels['macula'] == val]
    rate = sub['increased_cup_disc'].mean()
    print(f'  macula={val}: {len(sub)} images, glaucoma rate = {rate:.1%}')
mi2 = mutual_info_score(labels['macula'].astype(str), labels['increased_cup_disc'])
print(f'  Mutual info (macula vs glaucoma): {mi2:.4f}')

# 3. diabetes_time_y
print('\n── diabetes_time_y ──')
print(f'dtype: {labels["diabetes_time_y"].dtype}')
vals = labels['diabetes_time_y']
print(f'Non-null: {vals.notna().sum()} / {len(vals)}')
print(f'Sample values: {vals.dropna().unique()[:20]}')
# Convert to numeric
vals_num = pd.to_numeric(vals, errors='coerce')
print(f'Numeric non-null: {vals_num.notna().sum()}')
print(f'Mean: {vals_num.mean():.1f}, Median: {vals_num.median():.1f}, Max: {vals_num.max():.1f}')
# Check if NaN means no diabetes
no_diabetes = labels[labels['diabetes'] == 'no']
has_diabetes = labels[labels['diabetes'] == 'yes']
print(f'\n  diabetes=no  → diabetes_time_y NaN: {pd.to_numeric(no_diabetes["diabetes_time_y"], errors="coerce").isna().sum()}/{len(no_diabetes)}')
print(f'  diabetes=yes → diabetes_time_y NaN: {pd.to_numeric(has_diabetes["diabetes_time_y"], errors="coerce").isna().sum()}/{len(has_diabetes)}')
# Glaucoma rate by diabetes_time buckets
time_vals = pd.to_numeric(labels['diabetes_time_y'], errors='coerce')
labels['dm_time_num'] = time_vals
bins = [0, 5, 10, 15, 20, 100]
labels['dm_time_bin'] = pd.cut(labels['dm_time_num'], bins=bins, right=False)
for b in labels['dm_time_bin'].dropna().unique():
    sub = labels[labels['dm_time_bin'] == b]
    if len(sub) > 0:
        rate = sub['increased_cup_disc'].mean()
        print(f'  DM time {b}: n={len(sub)}, glaucoma={rate:.1%}')

# 4. insuline
print('\n── insuline ──')
print(f'dtype: {labels["insuline"].dtype}')
print(f'Values: {labels["insuline"].value_counts(dropna=False).to_dict()}')
for val in labels['insuline'].dropna().unique():
    sub = labels[labels['insuline'] == val]
    rate = sub['increased_cup_disc'].mean()
    print(f'  insuline={val}: {len(sub)} images, glaucoma rate = {rate:.1%}')
# NaN rate
nan_rate = labels['insuline'].isna().sum() / len(labels)
print(f'  NaN rate: {nan_rate:.1%}')
# For non-diabetic patients, insuline should be NaN
no_dm = labels[labels['diabetes'] == 'no']
has_dm = labels[labels['diabetes'] == 'yes']
print(f'  diabetes=no  → insuline NaN: {no_dm["insuline"].isna().sum()}/{len(no_dm)}')
print(f'  diabetes=yes → insuline NaN: {has_dm["insuline"].isna().sum()}/{len(has_dm)}')

# 5. vessels — might also be useful
print('\n── vessels ──')
print(f'Values: {labels["vessels"].value_counts().to_dict()}')
for val in sorted(labels['vessels'].unique()):
    sub = labels[labels['vessels'] == val]
    rate = sub['increased_cup_disc'].mean()
    print(f'  vessels={val}: {len(sub)} images, glaucoma rate = {rate:.1%}')

# 6. comorbidities — is this useful?
print('\n── comorbidities ──')
print(f'Unique values: {labels["comorbidities"].nunique()}')
print(f'NaN: {labels["comorbidities"].isna().sum()}')
print(f'\nTop 15:')
print(labels['comorbidities'].value_counts().head(15))

print()
print('='*80)
print('LEAKAGE RISK ASSESSMENT')
print('='*80)
print('''
optic_disc: VALUES are 1, 2, "bv"
  1 = normal optic disc (11,343 images) → 14.3% glaucoma
  2 = abnormal optic disc (2,910 images) → 42.2% glaucoma  
  "bv" = 1 image only
  
  ⚠️ HIGH CORRELATION with glaucoma (MI=0.06+)
  This is expected: increased cup-disc IS an optic disc abnormality.
  Question: Was optic_disc graded INDEPENDENTLY of increased_cup_disc?
  
  If same grader at same time → LABEL LEAKAGE (should NOT use)
  If independent assessment → VALID feature
  
  BRSET paper says: optic disc assessment is a separate grading variable.
  It captures overall disc appearance, not just cup-disc ratio.
  VERDICT: Moderate leakage risk, but adds genuine clinical info.

macula: Less correlated with glaucoma — safe to use.
  
diabetes_time_y: Numeric, NaN for non-diabetics → fill with 0. Safe.

insuline: Binary yes/no, only for diabetics → NaN for non-diabetics → fill with 0/no. Safe.
''')
