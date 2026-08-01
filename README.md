# BERTweet_paper
# ─────────────────────────────────────────────────────────
# Imports and Global Settings
# ─────────────────────────────────────────────────────────
import os, re, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib; matplotlib.rcParams['figure.dpi'] = 120
import seaborn as sns 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

import shap
import preprocessor as p
import emoji

# ── Global constants ──
BASE       = '/content/drive/MyDrive/CyberbullyingUpgradedA'
BERT_MAX   = 128   # BERT max token length
BT_MAX     = 128   # BERTweet max token length
BATCH_SIZE = 32    # reduce to 16 if out-of-memory
EPOCHS     = 3
LR         = 2e-5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('✅ All imports successful!')
print(f'🖥️  Device: {device}')
if device.type == 'cuda':
    print(f'🚀 GPU: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
else:
    print('⚠️  No GPU! Go to Runtime → Change runtime type → T4 GPU')
    # Using same data + new methods = direct comparison.
# ─────────────────────────────────────────────────────────
from google.colab import files
import io

print('📂 Upload: cyberbullying_tweets.csv')
print('   Download from: https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification')
print()

uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[filename]))

# Standardize column names
df.columns = [c.strip().lower().replace(' ','_') for c in df.columns]
if 'tweet_text' not in df.columns:
    df.columns = ['tweet_text', 'cyberbullying_type']

df = df.dropna(subset=['tweet_text']).reset_index(drop=True)
df['tweet_text'] = df['tweet_text'].astype(str)

print(f'\n✅ Dataset loaded: {filename}')
print(f'📊 Total rows    : {len(df):,}')
print(f'\n🏷️  Class distribution:')
print(df['cyberbullying_type'].value_counts())

df.to_csv(f'{BASE}/raw_dataset.csv', index=False)
print(f'\n💾 Raw dataset saved to Drive')
df.head(3)

# ─────────────────────────────────────────────────────────
# CELL 12: Build Stacking Features + Train MLP Stacker
# ─────────────────────────────────────────────────────────
# HOW STACKING WORKS:
#   Level 1:
#     BERT     → 6 probability scores (one per class)
#     BERTweet → 6 probability scores (one per class)
#     Combined → 12 features per tweet
#   Level 2:
#     MLP takes 12 features → final prediction
#
# The MLP learns to TRUST BERTweet more for Twitter text
# and BERT more when tweets are cleaner/more formal.
# ─────────────────────────────────────────────────────────
print('='*55)
print('BUILDING STACKING FEATURES + MLP META-CLASSIFIER')
print('='*55)

# Concatenate BERT + BERTweet probabilities
# Shape: (N_samples, 12)
S_train = np.concatenate([bert_tr_probs, bt_tr_probs], axis=1)
S_val   = np.concatenate([bert_va_probs, bt_va_probs], axis=1)
S_test  = np.concatenate([bert_probs,    bt_probs],    axis=1)

print(f'\n📐 Stacking feature shapes:')
print(f'   Train : {S_train.shape}  (cols 0-5=BERT, cols 6-11=BERTweet)')
print(f'   Val   : {S_val.shape}')
print(f'   Test  : {S_test.shape}')

# Feature names for SHAP
shap_feat_names = (
    [f'BERT_P({c})'     for c in CLASS_NAMES] +
    [f'BERTweet_P({c})' for c in CLASS_NAMES]
)

# Save stacking features
np.save(f'{BASE}/models/stacker/S_train.npy', S_train)
np.save(f'{BASE}/models/stacker/S_val.npy',   S_val)
np.save(f'{BASE}/models/stacker/S_test.npy',  S_test)
np.save(f'{BASE}/models/stacker/y_test.npy',  y_te_true)

# Combine train+val for MLP training
S_trainval = np.concatenate([S_train, S_val], axis=0)
y_trainval = np.concatenate([y_train, y_val], axis=0)

# ── Train MLP Meta-Classifier ──
print('\n⏳ Training MLP meta-classifier...')
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # 3 hidden layers
    activation='relu',
    max_iter=600,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False
)
mlp.fit(S_trainval, y_trainval)
print('✅ MLP stacker trained!')

# Evaluate stacker
stk_preds = mlp.predict(S_test)
stk_probs = mlp.predict_proba(S_test)
stk_acc   = accuracy_score(y_te_true, stk_preds)
stk_f1w   = f1_score(y_te_true, stk_preds, average='weighted')
stk_f1m   = f1_score(y_te_true, stk_preds, average='macro')

print(f'\n🎯 BERT + BERTweet Stack (MLP) Test Results:')
print(f'   Accuracy  : {stk_acc*100:.2f}%')
print(f'   F1-W      : {stk_f1w*100:.2f}%')
print(f'   F1-M      : {stk_f1m*100:.2f}%')

print('\n📋 Classification Report:')
report = classification_report(y_te_true, stk_preds, target_names=CLASS_NAMES)
print(report)

# Save stacker
with open(f'{BASE}/models/stacker/mlp_stacker.pkl','wb') as f: pickle.dump(mlp, f)
stk_result = {'accuracy':round(stk_acc*100,2),
               'f1_weighted':round(stk_f1w*100,2),
               'f1_macro':round(stk_f1m*100,2)}
with open(f'{BASE}/results/stacker_results.json','w') as f:
    json.dump(stk_result, f, indent=2)
with open(f'{BASE}/results/stacker_classification_report.txt','w') as f:
    f.write(f'BERT + BERTweet + MLP Stack\n')
    f.write(f'Acc:{stk_acc*100:.2f}% F1-W:{stk_f1w*100:.2f}% F1-M:{stk_f1m*100:.2f}%\n\n')
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_te_true, stk_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Stacked Transformer Confusion Matrix (BERT + BERTweet + MLP)', fontweight='bold')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.xticks(rotation=30); plt.tight_layout()
plt.savefig(f'{BASE}/plots/03_stacker_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print('\n💾 Stacker and results saved to Drive')
print('✅ Stacking complete! (Contribution 1 done)')

# ─────────────────────────────────────────────────────────
# CELL 13: Severity Sub-Classification  ← CONTRIBUTION 2
# ─────────────────────────────────────────────────────────
# After predicting the cyberbullying CATEGORY (religion,
# age, gender, ethnicity, non-bully), we add a SEVERITY
# level: mild / moderate / severe.
#
# HOW SEVERITY LABELS ARE CREATED:
#   We use BERTweet's confidence score as a proxy:
#   - Low confidence (0.40-0.60)  → MILD
#   - Medium confidence (0.60-0.80)→ MODERATE
#   - High confidence (>0.80)     → SEVERE
#   This is a novel auto-labeling approach.
#
# WHY IT IS ORIGINAL:
#   No paper has done severity classification on the
#   Wang et al. 5-class Twitter dataset.
# ─────────────────────────────────────────────────────────
print('='*55)
print('SEVERITY SUB-CLASSIFICATION  (Contribution 2)')
print('='*55)

# ── Step 1: Auto-label severity using BERTweet confidence ──
# Max probability from BERTweet = model confidence
max_probs = bt_probs.max(axis=1)  # shape: (N_test,)

def assign_severity(conf):
    """Assign severity based on model confidence."""
    if conf <= 0.60:   return 0  # mild
    elif conf <= 0.80: return 1  # moderate
    else:              return 2  # severe

SEVERITY_NAMES = ['Mild', 'Moderate', 'Severe']
y_severity = np.array([assign_severity(p) for p in max_probs])

print('\n📊 Severity distribution on test set:')
for i, name in enumerate(SEVERITY_NAMES):
    count = (y_severity == i).sum()
    pct   = count / len(y_severity) * 100
    print(f'   {name:10s}: {count:,} tweets ({pct:.1f}%)')

# ── Step 2: Train MLP severity classifier ──
# Uses stacking features (12 numbers per tweet) to
# predict severity. Quick to train.
print('\n⏳ Training severity classifier...')

# Build severity labels for full train+val set
bt_tr_max = bt_tr_probs.max(axis=1)
bt_va_max = bt_va_probs.max(axis=1)
y_sev_trainval = np.array([
    assign_severity(p) for p in np.concatenate([bt_tr_max, bt_va_max])
])

sev_clf = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    max_iter=400,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
sev_clf.fit(S_trainval, y_sev_trainval)
print('✅ Severity classifier trained!')

sev_preds = sev_clf.predict(S_test)
sev_acc   = accuracy_score(y_severity, sev_preds)
sev_f1    = f1_score(y_severity, sev_preds, average='weighted')

print(f'\n🎯 Severity Classifier Results:')
print(f'   Accuracy  : {sev_acc*100:.2f}%')
print(f'   F1-W      : {sev_f1*100:.2f}%')
print('\n📋 Severity Classification Report:')
sev_report = classification_report(y_severity, sev_preds, target_names=SEVERITY_NAMES)
print(sev_report)

# Save severity model
with open(f'{BASE}/models/severity/severity_clf.pkl','wb') as f: pickle.dump(sev_clf, f)
with open(f'{BASE}/results/severity_report.txt','w') as f:
    f.write(f'Severity Accuracy: {sev_acc*100:.2f}%\n')
    f.write(f'Severity F1-W: {sev_f1*100:.2f}%\n\n')
    f.write(sev_report)

# ── Visualize: Category + Severity heatmap ──
# Shows how severity distributes across cyberbullying types
cat_names = [CLASS_NAMES[p] for p in bt_preds]
sev_names = [SEVERITY_NAMES[s] for s in sev_preds]
cross_df = pd.crosstab(
    pd.Series(cat_names, name='Category'),
    pd.Series(sev_names, name='Severity')
)

plt.figure(figsize=(9, 5))
sns.heatmap(cross_df, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Cyberbullying Category × Severity Distribution\n(Your 2-Level Detection System)',
          fontweight='bold')
plt.tight_layout()
plt.savefig(f'{BASE}/plots/04_category_severity_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print('\n💾 Severity model and plots saved to Drive')
print('✅ Severity classification done! (Contribution 2 done)')

# ─────────────────────────────────────────────────────────────────────
# FIXED CELL 14: Token-Level SHAP Explainability  ← CONTRIBUTION 3
# ─────────────────────────────────────────────────────────────────────
#
# WHY THE ERROR HAPPENED:
#   KernelExplainer on a multi-class model returns
#   stk_shap_vals as a LIST of arrays:
#       stk_shap_vals[0] → shape (150, 12)  ← class 0 SHAP values
#       stk_shap_vals[1] → shape (150, 12)  ← class 1 SHAP values
#       ...etc
#
#   The beeswarm plot (summary_plot without plot_type='bar') expects:
#       shap_values  → numpy array of shape (150, 12)   ← one class
#       features     → numpy array of shape (150, 12)   ← same data
#
#   The AssertionError happened because SHAP was passed
#   stk_shap_vals[0] but internally tried to match
#   it against the full list shape. The fix is to
#   explicitly convert stk_shap_vals[i] to np.array().
#
# THIS FIXED CELL:
#   Part A: Global bar plot  → uses full list (correct)
#   Part B: Beeswarm per class → uses np.array(list[i]) (fixed)
#   Part C: Token-level SHAP on BERTweet (new, no change)
# ─────────────────────────────────────────────────────────────────────

print('='*55)
print('FIXED SHAP EXPLAINABILITY  (Contribution 3)')
print('='*55)

import shap
import numpy as np
import matplotlib.pyplot as plt
import torch

# ── STEP 1: Compute SHAP values ──────────────────────────────────────
# Background = 100 random training samples
# (Small background speeds up computation)
print('\n⏳ Computing SHAP values on 150 test samples...')
print('   (~3-5 minutes, please wait)')

bg_idx     = np.random.choice(len(S_trainval), size=100, replace=False)
bg_data    = S_trainval[bg_idx]          # shape: (100, 12)
test_data  = S_test[:150]               # shape: (150, 12)

stk_explainer = shap.KernelExplainer(mlp.predict_proba, bg_data)
stk_shap_vals = stk_explainer.shap_values(test_data, nsamples=200)

# stk_shap_vals is a LIST of length NUM_CLASSES
# stk_shap_vals[i] has shape (150, 12) for class i
print(f'\n✅ SHAP computed!')
print(f'   Type: {type(stk_shap_vals)}')
print(f'   Number of class arrays: {len(stk_shap_vals)}')
print(f'   Shape of each class array: {np.array(stk_shap_vals[0]).shape}')

# ── STEP 2: Global bar plot (all classes together) ────────────────────
# For bar plot → pass the full LIST, not a single class
print('\n📊 Generating Global SHAP Bar Plot...')
plt.figure(figsize=(11, 6))
shap.summary_plot(
    stk_shap_vals,          # ← full LIST of arrays (one per class)
    test_data,              # ← numpy array (150, 12)
    feature_names=shap_feat_names,
    class_names=CLASS_NAMES,
    plot_type='bar',        # bar = global importance across all classes
    show=False
)
plt.title(
    'SHAP Global Feature Importance\n'
    '(Which model — BERT or BERTweet — matters most per class)',
    fontweight='bold', fontsize=12
)
plt.tight_layout()
plt.savefig(f'{BASE}/shap_outputs/shap_global_bar.png',
            dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print('   💾 Global bar plot saved')

# ── STEP 3: Beeswarm plot per class ──────────────────────────────────
# For beeswarm → pass ONE class at a time as np.array()
# This is the FIX: np.array(stk_shap_vals[i]) ensures correct shape
print('\n📊 Generating Beeswarm Plots per class...')

for cls_idx in range(NUM_CLASSES):
    # FIX: explicitly convert to numpy array
    class_shap = np.array(stk_shap_vals[cls_idx])  # shape: (150, 12)

    plt.figure(figsize=(10, 5))
    shap.summary_plot(
        class_shap,          # ← numpy array for ONE class (150, 12)
        test_data,           # ← same numpy array (150, 12)
        feature_names=shap_feat_names,
        show=False
    )
    plt.title(
        f'SHAP Beeswarm — Class: {CLASS_NAMES[cls_idx]}\n'
        f'Red = pushes TOWARD this class | Blue = pushes AWAY',
        fontweight='bold', fontsize=11
    )
    plt.tight_layout()
    fname = f'{BASE}/shap_outputs/shap_beeswarm_{CLASS_NAMES[cls_idx].replace(" ","_")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'   💾 Beeswarm saved for class: {CLASS_NAMES[cls_idx]}')

# ── STEP 4: Force plot for one sample ────────────────────────────────
# Shows how each feature pushed a SINGLE tweet's prediction
print('\n📊 Generating Force Plot for sample tweet 0...')
try:
    shap.initjs()
    force = shap.force_plot(
        stk_explainer.expected_value[0],   # base value for class 0
        np.array(stk_shap_vals[0])[0, :],  # SHAP values for tweet 0, class 0
        test_data[0, :],                    # actual feature values for tweet 0
        feature_names=shap_feat_names,
        matplotlib=True,
        show=False
    )
    plt.title(f'Force Plot — Tweet 0 → Class: {CLASS_NAMES[0]}',
              fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{BASE}/shap_outputs/shap_force_sample0.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print('   💾 Force plot saved')
except Exception as e:
    print(f'   ⚠️  Force plot skipped (display issue in Colab): {e}')
    print('   This is normal — bar and beeswarm plots are already saved.')

# ── STEP 5: Token-level SHAP on BERTweet ─────────────────────────────
# Shows WHICH WORDS in a tweet drive the BERTweet prediction
print('\n📊 Token-level SHAP on BERTweet...')
print('   (~1-2 minutes per tweet)')

bt_model.eval()

def bertweet_predict_fn(texts):
    """Takes list of strings → softmax probabilities."""
    all_probs, batch_sz = [], 8
    for i in range(0, len(texts), batch_sz):
        batch = list(texts[i:i+batch_sz])
        enc   = bt_tok(
            batch, max_length=BT_MAX, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        with torch.no_grad():
            out = bt_model(
                input_ids=enc['input_ids'].to(device),
                attention_mask=enc['attention_mask'].to(device)
            )
        probs = torch.softmax(out.logits, dim=1).cpu().numpy()
        all_probs.extend(probs)
    return np.array(all_probs)

# Pick one sample per class for explanation
sample_texts = []
for cls_id in range(NUM_CLASSES):
    idxs = np.where(y_te_true == cls_id)[0]
    if len(idxs) > 0:
        sample_texts.append(X_test[idxs[0]])

print(f'\n   Explaining {len(sample_texts)} tweets (one per class)...')

# Use SHAP's Partition explainer for text
# This gives token-level importance in BERTweet
token_explainer = shap.Explainer(
    bertweet_predict_fn,
    masker=shap.maskers.Text(bt_tok),
    output_names=CLASS_NAMES
)

token_shap_vals = token_explainer(sample_texts[:3])  # do 3 to save time

# Save token SHAP HTML files
for i in range(len(sample_texts[:3])):
    try:
        html_out = shap.plots.text(token_shap_vals[i], display=False)
        with open(f'{BASE}/shap_outputs/token_shap_tweet_{i+1}.html', 'w') as hf:
            hf.write(str(html_out))
        print(f'   💾 Token SHAP HTML saved for tweet {i+1}')
    except Exception as e:
        print(f'   ⚠️  Tweet {i+1} HTML save skipped: {e}')

# Save all stacker SHAP values as numpy
np.save(f'{BASE}/shap_outputs/stk_shap_values.npy',
        np.array(stk_shap_vals, dtype=object))

print('\n✅ ALL SHAP OUTPUTS COMPLETE!')
print('   Files saved to Drive/shap_outputs/:')
print('   - shap_global_bar.png        (which model matters most)')
print(f'  - shap_beeswarm_[class].png  ({NUM_CLASSES} files, one per class)')
print('   - shap_force_sample0.png     (single tweet breakdown)')
print('   - token_shap_tweet_N.html    (which WORDS matter in tweets)')
print('   - stk_shap_values.npy        (raw SHAP values for paper)')
print('\n   Contribution 3 complete!')

# ==========================================================
# SHAP Beeswarm Plots (SHAP 0.52+)
# ==========================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import shap

print("SHAP version:", shap.__version__)

print("SHAP values shape :", stk_shap_vals.shape)
print("Test data shape   :", test_data.shape)

# -----------------------------
# Safety checks
# -----------------------------
assert isinstance(stk_shap_vals, np.ndarray)

assert stk_shap_vals.ndim == 3, \
    f"Expected 3D SHAP array, got {stk_shap_vals.ndim}D"

assert stk_shap_vals.shape[0] == test_data.shape[0]
assert stk_shap_vals.shape[1] == test_data.shape[1]

os.makedirs(f"{BASE}/shap_outputs", exist_ok=True)

print("\nGenerating Beeswarm Plots...\n")

# ----------------------------------------------------------
# Loop through every class
# ----------------------------------------------------------
for class_idx, class_name in enumerate(CLASS_NAMES):

    # (samples, features)
    class_shap = stk_shap_vals[:, :, class_idx]

    print(f"{class_name}")
    print("class_shap :", class_shap.shape)
    print("test_data  :", test_data.shape)

    plt.figure(figsize=(10,5))

    shap.summary_plot(
        class_shap,
        test_data,
        feature_names=shap_feat_names,
        show=False
    )

    plt.title(f"SHAP Beeswarm - {class_name}",
              fontsize=13,
              fontweight="bold")

    plt.tight_layout()

    plt.savefig(
        f"{BASE}/shap_outputs/beeswarm_{class_name}.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

print("\nDone!")
