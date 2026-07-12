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
