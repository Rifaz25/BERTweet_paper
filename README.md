## 1. Research Contributions

### Contribution 1 — BERT + BERTweet Stacking
BERT and BERTweet are fine-tuned independently. Each model produces six class probabilities. These are concatenated into a **12-dimensional stacking feature vector**, which is given to an MLP meta-classifier.

### Contribution 2 — Severity Sub-Classification
A secondary MLP predicts:

- Mild
- Moderate
- Severe

Severity is initially generated from BERTweet confidence as a weak-supervision proxy:

- confidence ≤ 0.60 → Mild
- 0.60 < confidence ≤ 0.80 → Moderate
- confidence > 0.80 → Severe

**Limitation:** the same confidence information is used in the severity pipeline, so the reported severity accuracy is affected by label circularity.

### Contribution 3 — Explainable AI
Two explainability approaches are used:

- **SHAP:** global and per-class importance of the 12 stacking features.
- **LIME:** local word/token explanations and aggregated global word importance.

---

## 2. Experimental Environment

The notebook was run in Google Colab with a **Tesla T4 GPU**.

Verified environment output:

```text
✅ All imports successful!
🖥️  Device: cuda
🚀 GPU: Tesla T4
   Memory: 15.6 GB
```

The notebook installed/used:

- transformers
- torch
- shap
- scikit-learn
- tweet-preprocessor
- emoji
- pandas
- numpy
- matplotlib
- seaborn
- imbalanced-learn
- lime

---

## 3. Dataset and Class Labels

The notebook uses the Wang et al. fine-grained Twitter cyberbullying dataset.

Verified preprocessing output reports:

- Original samples: **47,692**
- Missing values: **0**
- Duplicate rows: **36**
- After preprocessing: **47,126 tweets**
- Number of classes: **6**

Classes:

1. age
2. ethnicity
3. gender
4. not_cyberbullying
5. other_cyberbullying
6. religion

The verified split was:

| Split | Samples |
|---|---:|
| Train | 37,700 |
| Validation | 4,713 |
| Test | 4,713 |

---

# 4. Step-by-Step Verified Pipeline

## Step 1 — Install Libraries

This corresponds to the successful library-installation cell in the notebook.

```python
# CELL 2: Install All Libraries
print('📦 Installing libraries... (~3 minutes)')

!pip install transformers==4.40.0 -q
!pip install torch -q
!pip install shap -q
!pip install scikit-learn -q
!pip install tweet-preprocessor -q
!pip install emoji==1.7.0 -q
!pip install pandas numpy matplotlib seaborn -q
!pip install imbalanced-learn -q

print('✅ All libraries installed!')
```

Verified output ended with:

```text
✅ All libraries installed!
```

---

## Step 2 — Import Libraries and Check GPU

```python
print('Checking imports and device...')
# Use the imports from the notebook's import cell.
# The original successful cell also initializes the CUDA device.
```

Verified output:

```text
✅ All imports successful!
🖥️  Device: cuda
🚀 GPU: Tesla T4
   Memory: 15.6 GB
```

---

## Step 3 — Twitter-Specific Preprocessing

The verified preprocessing cell performs the notebook's Twitter cleaning pipeline.

It removes/normalizes platform-specific artifacts and saves the processed dataset.

Verified output:

```text
🔄 Preprocessing all tweets...
   Removed 566 empty tweets → 47,126 remaining

📋 Before vs After (3 examples):
─────────────────────────────────────────────────────────────────
BEFORE: In other words #katandandre, your food was crapilicious! #mkr
AFTER : in other words katandandre your food was crapilicious mkr

BEFORE: Why is #aussietv so white? #MKR #theblock #ImACelebrityAU #today #sunrise #studio10 #
AFTER : why is aussietv so white mkr theblock imacelebrityau today sunrise studio10 neighbour

BEFORE: @XochitlSuckkks a classy whore? Or more red velvet cupcakes?
AFTER : a classy whore or more red velvet cupcakes

💾 Preprocessed dataset saved to Drive
✅ Preprocessing complete!
```

---

## Step 4 — Label Encoding and Data Split

The verified notebook encoding is:

```text
0 → age
1 → ethnicity
2 → gender
3 → not_cyberbullying
4 → other_cyberbullying
5 → religion
```

Verified split:

```text
Train : 37,700 (80%)
Val   : 4,713 (10%)
Test  : 4,713 (10%)
```

---

# 5. Traditional ML Baselines

The base-paper-style baselines use TF-IDF features.

Verified TF-IDF shape:

```text
(37700, 50000)
```

### Verified baseline results

| Model | Accuracy | F1-Weighted | F1-Macro |
|---|---:|---:|---:|
| Random Forest | 80.88% | 81.16% | 80.77% |
| SVM (LinearSVC) | 81.99% | 81.94% | 81.55% |
| Logistic Regression | 82.54% | 82.60% | 82.25% |
| Naive Bayes | 76.43% | 74.33% | 73.97% |
| KNN | 21.94% | 18.61% | 18.69% |

---

# 6. Fine-Tune BERT

BERT uses `bert-base-uncased` and is fine-tuned for the six-class classification task.

The verified notebook used **3 epochs**.

Training:

```text
Epoch 1/3
Train → Loss:0.6111  Acc:76.60%
Val   → Loss:0.3512  Acc:85.47%

Epoch 2/3
Train → Loss:0.3097  Acc:87.64%
Val   → Loss:0.3373  Acc:86.29%

Epoch 3/3
Train → Loss:0.2364  Acc:90.86%
Val   → Loss:0.3537  Acc:86.55%
```

Best validation accuracy:

```text
86.55%
```

### BERT test result

```text
Accuracy : 85.72%
F1-W     : 85.65%
F1-M     : 85.36%
```

---

# 7. Fine-Tune BERTweet

BERTweet is the Twitter-domain transformer component.

The notebook reports:

```text
Model: vinai/bertweet-base
Pre-trained on: 850 million English tweets
Vocab size: 64,000 tokens
```

The verified notebook again used **3 epochs**.

Training:

```text
Epoch 1/3
Train → Loss:0.6285  Acc:77.44%
Val   → Loss:0.3498  Acc:86.44%

Epoch 2/3
Train → Loss:0.3174  Acc:87.57%
Val   → Loss:0.3202  Acc:86.63%

Epoch 3/3
Train → Loss:0.2533  Acc:90.35%
Val   → Loss:0.3243  Acc:87.44%
```

Best validation accuracy:

```text
87.44%
```

### BERTweet test result

```text
Accuracy : 86.89%
F1-W     : 86.70%
F1-M     : 86.44%
```

---

# 8. BERT + BERTweet MLP Stacking — Contribution 1

Each transformer generates six class probabilities:

```text
BERT     → 6 probabilities
BERTweet → 6 probabilities
```

They are concatenated:

```text
6 + 6 = 12 stacking features
```

Verified shapes:

```text
Train : (37700, 12)
Val   : (4713, 12)
Test  : (4713, 12)
```

The MLP meta-classifier uses three hidden layers:

```text
(256, 128, 64)
```

with ReLU activation and early stopping.

### Verified stack results

```text
Accuracy : 86.74%
F1-W     : 86.58%
F1-M     : 86.32%
```

### Classification report

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| age | 0.98 | 0.98 | 0.98 |
| ethnicity | 0.98 | 0.98 | 0.98 |
| gender | 0.88 | 0.89 | 0.88 |
| not_cyberbullying | 0.73 | 0.59 | 0.65 |
| other_cyberbullying | 0.66 | 0.78 | 0.72 |
| religion | 0.96 | 0.97 | 0.97 |

---

# 9. Severity Sub-Classification — Contribution 2

Severity is generated from the maximum BERTweet probability.

```python
def assign_severity(conf):
    if conf <= 0.60:
        return 0       # Mild
    elif conf <= 0.80:
        return 1       # Moderate
    else:
        return 2       # Severe
```

Verified test distribution:

```text
Mild      : 276 tweets (5.9%)
Moderate  : 598 tweets (12.7%)
Severe    : 3,839 tweets (81.5%)
```

Verified classifier result:

```text
Accuracy : 99.26%
F1-W     : 99.26%
```

### Important interpretation

This **99.26% should not be presented as an unbiased severity result**.

The severity labels are derived from BERTweet confidence, and confidence is also part of the information used by the severity classifier. This creates **label circularity**.

---

# 10. SHAP Explainability — Contribution 3

The corrected SHAP implementation uses SHAP 0.52.0.

Verified shapes:

```text
SHAP values shape : (150, 12, 6)
Test data shape   : (150, 12)
```

Interpretation:

- 150 test tweets were explained.
- 12 stacking features were analyzed.
- 6 output classes were considered.

For each class, the correct slice is:

```python
class_shap = stk_shap_vals[:, :, class_idx]
```

which gives:

```text
(150, 12)
```

This fixes the earlier shape-mismatch problem.

The notebook successfully generated:

- Global SHAP feature importance
- Per-class SHAP beeswarm plots

---

# 11. LIME Explainability

## 11.1 Install LIME

```python
!pip install lime
```

Verified:

```text
Successfully installed lime-0.2.0.1
```

## 11.2 BERTweet Prediction Wrapper

LIME needs a function that returns probabilities for perturbed text.

```python
import torch
import numpy as np

def bertweet_predict_proba(texts):
    bt_model.eval()
    all_probs = []
    batch_size = 16

    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i+batch_size])

        enc = bt_tok(
            batch,
            max_length=BT_MAX,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            out = bt_model(
                input_ids=enc['input_ids'].to(device),
                attention_mask=enc['attention_mask'].to(device)
            )

        probs = torch.softmax(out.logits, dim=1).cpu().numpy()
        all_probs.extend(probs)

    return np.array(all_probs)

test_output = bertweet_predict_proba(
    ["you are so old nobody wants you here"]
)

print('✅ Wrapper function works!')
print(f'Output shape: {test_output.shape}')
print(f'Probabilities: {test_output[0].round(3)}')
```

Verified output:

```text
✅ Wrapper function works!
Output shape: (1, 6)
Probabilities: [0.161 0.152 0.153 0.183 0.159 0.191]
```

## 11.3 Create LIME Explainer

```python
from lime.lime_text import LimeTextExplainer

lime_explainer = LimeTextExplainer(
    class_names=CLASS_NAMES,
    random_state=42
)

print('✅ LIME text explainer created!')
print(f'Classes: {CLASS_NAMES}')
```

Verified output:

```text
✅ LIME text explainer created!
Classes: ['age', 'ethnicity', 'gender',
          'not_cyberbullying',
          'other_cyberbullying',
          'religion']
```

## 11.4 Generate One LIME Explanation per Class

The verified notebook uses:

```python
explanation = lime_explainer.explain_instance(
    sample_text,
    bertweet_predict_proba,
    num_features=10,
    num_samples=500,
    labels=[np.argmax(probs)]
)
```

It saves each explanation as HTML and stores the results in JSON.

Example from the verified output:

```text
True label: not_cyberbullying
Predicted: not_cyberbullying (confidence: 18.8%)

Top words:
t            : -0.0037 (AWAY FROM)
didn         : +0.0020 (TOWARD)
of           : +0.0019 (TOWARD)
see          : +0.0014 (TOWARD)
single       : +0.0013 (TOWARD)
what         : +0.0012 (TOWARD)
```

The notebook completed:

```text
💾 All LIME explanations saved to Drive (HTML + JSON)
✅ Individual tweet explanations complete!
```

> Some individual predictions in the recorded LIME run had low confidence. They should be treated as qualitative explanation examples rather than strong evidence of classification performance.

---

# 12. Global LIME Word Importance

The notebook also aggregates LIME importance across **30 test tweets**.

Verified top global words included:

```text
i        : 0.020
a        : 0.015
you      : 0.014
to       : 0.009
but      : 0.008
in       : 0.007
not      : 0.007
and      : 0.006
t        : 0.006
with     : 0.005
school   : 0.005
my       : 0.005
this     : 0.004
that     : 0.004
mkr      : 0.004
```

The notebook successfully generated:

```text
lime_global_word_importance.png
lime_global_importance.json
```

---

# 13. SHAP vs LIME

The notebook includes a qualitative cross-check between SHAP and LIME.

For one sample, the recorded LIME top words were:

```text
baby
got
her
high
husband
in
pregnant
that
their
yep
```

The notebook notes that the corresponding SHAP token output should be compared manually to calculate overlap.

This is intended as a **qualitative cross-method sanity check**, not as a primary quantitative metric.

---

# 14. Full Pipeline Demo

The verified notebook demonstrates the complete two-level prediction pipeline.

Example input:

```text
"You're too old to be on social media, grandma. Nobody wants you here."
```

After preprocessing:

```text
"you're too old to be on social media grandma nobody wants you here"
```

### Level 1 — Category

```text
Prediction: OTHER_CYBERBULLYING
Confidence: 37.6%
```

### Level 2 — Severity

```text
Prediction: MODERATE
BERTweet confidence: 68.1%
Severity rule: Medium confidence → Moderate
```

Final:

```text
Tweet Type : other_cyberbullying
Severity   : Moderate
```

---

# 15. Final Results

| Model | Type | Accuracy | F1-Weighted | F1-Macro |
|---|---|---:|---:|---:|
| Random Forest | Baseline | 80.88 | 81.16 | 80.77 |
| SVM (LinearSVC) | Baseline | 81.99 | 81.94 | 81.55 |
| Logistic Regression | Baseline | 82.54 | 82.60 | 82.25 |
| Naive Bayes | Baseline | 76.43 | 74.33 | 73.97 |
| KNN | Baseline | 21.94 | 18.61 | 18.69 |
| BERT | Proposed | 85.72 | 85.65 | 85.36 |
| **BERTweet** | **Proposed** | **86.89** | **86.70** | **86.44** |
| Stacked BERT+BERTweet+MLP | Proposed | 86.74 | 86.58 | 86.32 |

### Main result

**BERTweet achieved the highest raw accuracy: 86.89%.**

The stacked model achieved **86.74%**, slightly below BERTweet alone, but the stack provides the framework used for the paper's severity and explainability components.
