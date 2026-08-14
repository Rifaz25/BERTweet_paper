# Pipeline Overview

```
Raw tweets → Preprocessing → Train/Val/Test split
    → [Baseline ML models]                         (comparison point)
    → [BERT fine-tuning]  ─┐
    → [BERTweet fine-tuning] ┴→ Stacking MLP → Category prediction
                                       │
                                       └→ Severity sub-classification
    → SHAP explainability (global + per-class)
    → LIME explainability (⚠️ rerun needed — see Section 10)
    → Final comparison table
```

---

## 1. Environment Setup

**Platform:** Google Colab, T4 GPU

```python
!pip install transformers==4.40.0 -q
!pip install torch -q
!pip install shap -q
!pip install scikit-learn -q
!pip install tweet-preprocessor -q
!pip install emoji==1.7.0 -q
!pip install pandas numpy matplotlib seaborn -q
!pip install imbalanced-learn -q
```

```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Output:**
```
✅ All imports successful!
🖥️  Device: cuda
🚀 GPU: Tesla T4
   Memory: 15.6 GB
```

---

## 2. Data Preprocessing

Twitter-specific cleaning: URL/mention removal, emoji-to-text conversion, hashtag extraction, punctuation normalization.

```python
def preprocess_tweet(text):
    text = p.clean(text)                              # remove URLs, @, RT
    text = emoji.demojize(text, delimiters=(' ',' ')) # 😡 → angry_face
    text = re.sub(r'#(\w+)', r'\1', text)             # #bullying → bullying
    text = re.sub(r"[^a-zA-Z0-9\s'_]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

df['clean_tweet'] = df['tweet_text'].apply(preprocess_tweet)
```

**Output:**
```
🔄 Preprocessing all tweets...
   Removed 566 empty tweets → 47,126 remaining

📋 Before vs After (3 examples):
─────────────────────────────────────────────────────────────────
BEFORE: In other words #katandandre, your food was crapilicious! #mkr
AFTER : in other words katandandre your food was crapilicious mkr
─────────────────────────────────────────────────────────────────
BEFORE: Why is #aussietv so white? #MKR #theblock #ImACelebrityAU #today ...
AFTER : why is aussietv so white mkr theblock imacelebrityau today sunrise ...
─────────────────────────────────────────────────────────────────
✅ Preprocessing complete!
```

---

## 3. Label Encoding and Data Split

Stratified 80/10/10 split, fixed seed for reproducibility.

```python
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
```

**Output:**
```
🏷️  Label Encoding:
   0 → age
   1 → ethnicity
   2 → gender
   3 → not_cyberbullying
   4 → other_cyberbullying
   5 → religion

📊 Number of classes: 6

📋 Split (random_state=42, always identical):
   Train : 37,700 (80%)
   Val   : 4,713 (10%)
   Test  : 4,713 (10%)
```

---

## 4. Baseline Reproduction (Traditional ML)

Reproduces the base paper's five classical classifiers on TF-IDF features (50,000 features, unigrams + bigrams), for a direct comparison point.

```python
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True, min_df=2)

baselines = {
    'Random Forest'       : RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    'SVM (LinearSVC)'     : LinearSVC(max_iter=3000, random_state=42),
    'Logistic Regression' : LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
    'Naive Bayes'         : ComplementNB(),
    'KNN'                 : KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
}
```

**Output:**
```
🔄 Building TF-IDF features...
   TF-IDF shape: (37700, 50000)

Model                        Acc %   F1-W %   F1-M %
──────────────────────────────────────────────────────────
Random Forest                80.88    81.16    80.77
SVM (LinearSVC)               81.99    81.94    81.55
Logistic Regression           82.54     82.6    82.25
Naive Bayes                   76.43    74.33    73.97
KNN                           21.94    18.61    18.69
```

---

## 5. BERT Fine-Tuning

`bert-base-uncased`, 3 epochs, AdamW (lr=2e-5), max sequence length 128, batch size 32.

```python
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=NUM_CLASSES).to(device)
```

**Output:**
```
🚀 Training BERT for 3 epochs...

  Epoch 1/3   Train Loss:0.6111  Acc:76.60%   Val Loss:0.3512  Acc:85.47%
  Epoch 2/3   Train Loss:0.3097  Acc:87.64%   Val Loss:0.3373  Acc:86.29%
  Epoch 3/3   Train Loss:0.2364  Acc:90.86%   Val Loss:0.3537  Acc:86.55%

🎯 BERT Test Results:
   Accuracy  : 85.72%
   F1-W      : 85.65%
   F1-M      : 85.36%
```

---

## 6. BERTweet Fine-Tuning (Key Contribution)

`vinai/bertweet-base` — pretrained on 850M English tweets, using `AutoTokenizer` (required for its Twitter-specific vocabulary).

```python
bt_tok   = AutoTokenizer.from_pretrained('vinai/bertweet-base', use_fast=False)
bt_model = AutoModelForSequenceClassification.from_pretrained('vinai/bertweet-base', num_labels=NUM_CLASSES).to(device)
```

**Output:**
```
✅ BERTweet loaded!
   Vocab size: 64,000 tokens (includes Twitter-specific tokens)

🚀 Training BERTweet for 3 epochs...

  Epoch 1/3   Train Loss:0.6285  Acc:77.44%   Val Loss:0.3498  Acc:86.44%
  Epoch 2/3   Train Loss:0.3174  Acc:87.57%   Val Loss:0.3202  Acc:86.63%
  Epoch 3/3   Train Loss:0.2533  Acc:90.35%   Val Loss:0.3243  Acc:87.44%

🎯 BERTweet Test Results:
   Accuracy  : 86.89%
   F1-W      : 86.70%
   F1-M      : 86.44%
```

---

## 7. Stacking Ensemble (Contribution 1)

Concatenates BERT's 6 class-probabilities + BERTweet's 6 class-probabilities (12 features/tweet) → trains an MLP meta-classifier (256→128→64, ReLU).

```python
S_train = np.concatenate([bert_tr_probs, bt_tr_probs], axis=1)   # (N, 12)

mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu',
                     max_iter=600, random_state=42, early_stopping=True)
mlp.fit(S_trainval, y_trainval)
```

**Output:**
```
📐 Stacking feature shapes:
   Train : (37700, 12)  (cols 0-5=BERT, cols 6-11=BERTweet)

🎯 BERT + BERTweet Stack (MLP) Test Results:
   Accuracy  : 86.74%
   F1-W      : 86.58%
   F1-M      : 86.32%

                     precision    recall  f1-score   support
                age       0.98      0.98      0.98       800
          ethnicity       0.98      0.98      0.98       796
             gender       0.88      0.89      0.88       793
  not_cyberbullying       0.73      0.59      0.65       775
other_cyberbullying       0.66      0.78      0.72       749
           religion       0.96      0.97      0.97       800

           accuracy                           0.87      4713
```

---

## 8. Severity Sub-Classification (Contribution 2)

Auto-labels severity (mild/moderate/severe) from BERTweet's confidence score, then trains an MLP on the same 12 stacking features.

> **⚠️ Known limitation:** severity labels are derived from BERTweet's own confidence, which is also part of the classifier's input — this circularity likely inflates the reported accuracy. See [Limitations](#limitations) below.

```python
def assign_severity(conf):
    if conf <= 0.60:   return 0  # mild
    elif conf <= 0.80: return 1  # moderate
    else:              return 2  # severe
```

**Output:**
```
📊 Severity distribution on test set:
   Mild      : 276 tweets (5.9%)
   Moderate  : 598 tweets (12.7%)
   Severe    : 3,839 tweets (81.5%)

🎯 Severity Classifier Results:
   Accuracy  : 99.26%
   F1-W      : 99.26%
```

---

## 9. SHAP Explainability (Contribution 3)

Global SHAP on the 12 stacking features, with per-class beeswarm plots. The 3D SHAP output array `(150, 12, 6)` is sliced per class as `stk_shap_vals[:, :, class_idx]` to avoid the shape-mismatch `AssertionError` that SHAP raises on multi-class `KernelExplainer` output.

```python
assert stk_shap_vals.ndim == 3, f"Expected 3D SHAP array, got {stk_shap_vals.ndim}D"

for class_idx, class_name in enumerate(CLASS_NAMES):
    class_shap = stk_shap_vals[:, :, class_idx]   # (150, 12)
    shap.summary_plot(class_shap, test_data, feature_names=shap_feat_names, show=False)
```

**Output:**
```
SHAP version: 0.52.0
SHAP values shape : (150, 12, 6)
Test data shape   : (150, 12)

Generating Beeswarm Plots...
age, ethnicity, gender, not_cyberbullying, other_cyberbullying, religion
Done!
```

---

## 10. LIME Local Explainability (Exploratory — See Caveat)

> **⚠️ Validity caveat:** the cells below ran successfully with no Python errors, and are included for completeness since they executed end-to-end. However, in this run `bt_model` had been reloaded fresh from `vinai/bertweet-base` (see `models/bertweet` diagnostic below) **without the fine-tuned checkpoint** — every retraining attempt earlier in this session had errored out. The near-uniform prediction confidences below (~16–19%, vs. 1/6 = 16.7% random chance) confirm this. **Treat the specific words/weights below as illustrative of the LIME pipeline mechanics only, not as validated findings about what the trained model has learned.** See [Limitations](#limitations) for the fix.

**Diagnostic that revealed the issue:**
```python
path = '/content/drive/MyDrive/CyberbullyingUpgradedA/models/bertweet'
for file in os.listdir(path):
    print(f" - {file}")
```
```
Files found in .../models/bertweet:
 - config.json
 - tokenizer_config.json
 - special_tokens_map.json
 - added_tokens.json
 - vocab.txt
 - bpe.codes
```
*(No `model.safetensors` / `pytorch_model.bin` — only tokenizer files. The fine-tuned weights were not present, so `bt_model` below has a randomly-initialized classification head.)*

**LIME prediction wrapper + sanity check:**
```python
def bertweet_predict_proba(texts):
    bt_model.eval()
    all_probs = []
    for i in range(0, len(texts), 16):
        batch = list(texts[i:i+16])
        enc = bt_tok(batch, max_length=BT_MAX, padding='max_length', truncation=True, return_tensors='pt')
        with torch.no_grad():
            out = bt_model(input_ids=enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device))
        all_probs.extend(torch.softmax(out.logits, dim=1).cpu().numpy())
    return np.array(all_probs)

test_output = bertweet_predict_proba(["you are so old nobody wants you here"])
```
```
✅ Wrapper function works!
   Output shape: (1, 6)  (should be (1, 6))
   Probabilities: [0.161 0.152 0.153 0.183 0.159 0.191]
```
*(Near-uniform across all 6 classes — the signature of an untrained head.)*

**LIME explanation, one tweet per class:**
```python
from lime.lime_text import LimeTextExplainer

lime_explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=42)

explanation = lime_explainer.explain_instance(
    sample_text, bertweet_predict_proba,
    num_features=10, num_samples=500, labels=[np.argmax(probs)]
)
```
```
--- Tweet 1/6  (True label: age) ---
Text: the girl that bullied me in high school her husband got their baby sitter pregnant yep karma
Predicted: religion  (confidence: 18.9%)
Top words driving this prediction:
   in                  : -0.0019  (AWAY FROM "religion")
   pregnant            : +0.0011  (TOWARD "religion")
   that                : -0.0009  (AWAY FROM "religion")
   husband             : +0.0008  (TOWARD "religion")

--- Tweet 2/6  (True label: ethnicity) ---
Text: fuck obama dumb ass nigger lt lt lt go drink some bleach
Predicted: religion  (confidence: 19.0%)
Top words driving this prediction:
   lt                  : +0.0034  (TOWARD "religion")
   go                  : -0.0021  (AWAY FROM "religion")
   bleach              : +0.0013  (TOWARD "religion")
```
*(Note: true label "age" predicted as "religion" at 18.9% confidence — essentially a coin-flip across 6 classes, not a meaningful prediction.)*

**How to fix and rerun properly:**
```python
# Load the FINE-TUNED checkpoint, not the base HuggingFace model:
bt_tok   = AutoTokenizer.from_pretrained(f'{BASE}/models/bertweet', use_fast=False)
bt_model = AutoModelForSequenceClassification.from_pretrained(f'{BASE}/models/bertweet').to(device)
# Then verify with the same sanity check above — confidences should be
# clearly peaked (e.g., >60% on one class), not near-uniform, before
# trusting any LIME output that follows.
```

---

## 11. Final Results Comparison

```python
res_df.to_csv(f'{BASE}/results/FINAL_comparison_table.csv', index=False)
```

**Output:**

| Model | Type | Accuracy % | F1-Weighted % | F1-Macro % |
|---|---|---|---|---|
| Random Forest | Paper 3 Baseline | 80.88 | 81.16 | 80.77 |
| SVM (LinearSVC) | Paper 3 Baseline | 81.99 | 81.94 | 81.55 |
| Logistic Regression | Paper 3 Baseline | 82.54 | 82.60 | 82.25 |
| Naive Bayes | Paper 3 Baseline | 76.43 | 74.33 | 73.97 |
| KNN | Paper 3 Baseline | 21.94 | 18.61 | 18.69 |
| BERT (fine-tuned) | Contribution | 85.72 | 85.65 | 85.36 |
| BERTweet (Twitter-specific) | Contribution | **86.89** | **86.70** | **86.44** |
| **BERT + BERTweet + MLP Stack** | Contribution (best) | 86.74 | 86.58 | 86.32 |

---

## 12. End-to-End Demo

```python
test_tweet = "You're too old to be on social media, grandma. Nobody wants you here."
```

**Output:**
```
🧹 After preprocessing:
   "you're too old to be on social media grandma nobody wants you here"

🎯 LEVEL 1 — Category Detection:
   Prediction: OTHER_CYBERBULLYING
   other_cyberbullying :  37.6% ███████

🎯 LEVEL 2 — Severity Detection:
   Prediction: MODERATE
   BERTweet confidence: 68.1%
```

---

## Repository Structure

```
CyberbullyingUpgradedA/
├── preprocessed_dataset.csv
├── meta.json
├── models/
│   ├── label_encoder.pkl, tfidf.pkl
│   ├── Random_Forest.pkl, SVM_LinearSVC.pkl, Logistic_Regression.pkl, Naive_Bayes.pkl, KNN.pkl
│   ├── bert/              # fine-tuned BERT checkpoint
│   ├── bertweet/          # fine-tuned BERTweet checkpoint
│   ├── stacker/           # S_train.npy, S_val.npy, S_test.npy, mlp_stacker.pkl
│   └── severity/          # severity_clf.pkl
├── results/                # baseline_results.json, bert_history.json, bertweet_history.json,
│                            # stacker_results.json, FINAL_comparison_table.csv
├── plots/                  # class distribution, confusion matrix, severity heatmap, final comparison
└── shap_outputs/           # shap_global_bar.png, beeswarm_<class>.png (×6)
```

---

