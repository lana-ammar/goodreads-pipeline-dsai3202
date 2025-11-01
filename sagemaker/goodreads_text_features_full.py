#!/usr/bin/env python3
# --------------------------------------------------------------
# goodreads_text_features_full.py
# FULLY WORKING – CPU, SKLearnProcessor, features_v2/
# --------------------------------------------------------------

#!/usr/bin/env python3
# --------------------------------------------------------------
# goodreads_text_features_full.py
# --------------------------------------------------------------

import os
import sys
import subprocess

print("\n=== STARTING FULL FEATURE EXTRACTION JOB ===", flush=True)

# ------------------- INSTALL DEPENDENCIES -------------------
def install_requirements():
    req_path = "/opt/ml/processing/input/requirements/requirements.txt"
    fallback_packages = [
        "textblob", "transformers", "torch", "nltk", "emoji",
        "scikit-learn", "pandas", "pyarrow"
    ]
    try:
        if os.path.exists(req_path):
            print(f"Installing from: {req_path}", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
        else:
            print("No requirements.txt found. Installing fallback packages...", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", *fallback_packages])
        print("✅ Dependencies installed successfully", flush=True)
    except Exception as e:
        print(f"❌ Failed to install dependencies: {e}", flush=True)
        sys.exit(1)

install_requirements()

# ------------------- CONTINUE IMPORTS AFTER INSTALL -------------------
import gc
import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import torch
from transformers import AutoTokenizer, AutoModel
import emoji
from pathlib import Path

print("✅ All imports successful. Proceeding with feature extraction...", flush=True)

# ------------------- INSTALL DEPENDENCIES FIRST -------------------
def install_requirements():
    req_file = "/opt/ml/processing/input/requirements/requirements.txt"
    fallback_packages = [
        "textblob", "transformers", "torch", "nltk", "emoji",
        "scikit-learn", "pandas", "pyarrow"
    ]
    try:
        if os.path.exists(req_file):
            print("Installing from requirements.txt...", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        else:
            print("requirements.txt not found. Installing fallback packages...", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", *fallback_packages])
        print("Dependencies installed successfully.", flush=True)
    except Exception as e:
        print(f"Failed to install dependencies: {e}", flush=True)
        sys.exit(1)

install_requirements()

# ------------------- NOW IMPORT THE REST -------------------
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import torch
from transformers import AutoTokenizer, AutoModel
import emoji

print("All imports successful. Starting feature engineering...", flush=True)

# ------------------- NLTK DOWNLOADS -------------------
nltk.download('vader_lexicon', quiet=True)
nltk.download('opinion_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

# ------------------- PATHS & ENV -------------------
INPUT_DIR  = os.getenv("INPUT_DIR",  "/opt/ml/processing/input/features")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/opt/ml/processing/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LEN    = int(os.getenv("MAX_LEN", "96"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))  # CPU-friendly
ROW_CHUNK  = os.getenv("ROW_CHUNK", None)

# ------------------- FIND PARQUET FILES -------------------
def find_parquet_files(root):
    exts = (".parquet", ".parquet.snappy")
    return [os.path.join(dp, f) for dp, _, fn in os.walk(root)
            for f in fn if f.lower().endswith(exts)]

parquet_files = find_parquet_files(INPUT_DIR)
if not parquet_files:
    raise FileNotFoundError(f"No parquet files found in {INPUT_DIR}")
print(f"Found {len(parquet_files)} parquet file(s)", flush=True)

# ------------------- LOAD DATA -------------------
dfs = []
total_rows = 0
for fp in parquet_files:
    part = pd.read_parquet(fp)
    dfs.append(part)
    total_rows += len(part)
    print(f"Loaded {len(part)} rows from {os.path.basename(fp)} (total: {total_rows})", flush=True)
    if ROW_CHUNK and total_rows >= int(ROW_CHUNK):
        break

df = pd.concat(dfs, ignore_index=True)
if ROW_CHUNK:
    df = df.head(int(ROW_CHUNK))
print(f"Final DataFrame: {len(df):,} rows × {len(df.columns)} cols", flush=True)

# ------------------- ENSURE REQUIRED COLUMNS -------------------
REQ_COLS = [
    "book_id", "user_id", "review_id", "rating", "rating_count",
    "review_text", "review_length_raw", "review_char_count",
    "date_added", "n_votes", "title", "publication_year", "author_names"
]

for col in REQ_COLS:
    if col not in df.columns:
        df[col] = pd.NA
df = df[REQ_COLS]

# ------------------- TEXT COLUMN -------------------
txt = df["review_text"].fillna("")

# ------------------- HELPER REGEX -------------------
WORD_RE  = re.compile(r"\w+")
EMOJI_RE = re.compile("[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F"
                      "\U0001F680-\U0001F6FF\U0001F700-\U0001F77F"
                      "\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
                      "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F"
                      "\U0001FA70-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]", flags=re.UNICODE)

# ------------------- FEATURE FUNCTIONS -------------------
def formatting_features(df, txt):
    df["sentence_count"]     = txt.str.count(r"[.!?]")
    df["paragraph_count"]    = txt.str.count(r"\n") + 1
    df["quote_count"]        = txt.str.count('"')
    df["exclamation_count"]  = txt.str.count('!')
    df["question_count"]     = txt.str.count(r"\?")

def length_features(df, txt):
    df["punct_count"] = txt.apply(lambda s: sum(ch in string.punctuation for ch in s))
    df["caps_count"]  = txt.apply(lambda s: sum(ch.isupper() for ch in s))
    df["emoji_count"] = txt.apply(lambda s: len(EMOJI_RE.findall(s)))

def lexical_features(df, txt):
    df["lexical_diversity"] = txt.apply(
        lambda s: len(set(s.split())) / max(1, len(s.split()))
    )

def readability_features(df, txt):
    def flesch_kincaid(t):
        if not t.strip(): return 0.0
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        words = WORD_RE.findall(t)
        if not sents or not words: return 0.0
        syllables = sum(len(re.findall(r"[aeiouyAEIOUY]+", w)) for w in words)
        return 0.39 * (len(words) / len(sents)) + 11.8 * (syllables / len(words)) - 15.59
    df["readability_grade"] = txt.apply(flesch_kincaid).clip(lower=0)

def sentiment_features(df, txt):
    sia = SentimentIntensityAnalyzer()
    scores = txt.apply(lambda x: sia.polarity_scores(x) if pd.notna(x) else {})
    df["vader_pos"] = scores.apply(lambda d: d.get("pos", 0))
    df["vader_neg"] = scores.apply(lambda d: d.get("neg", 0))
    df["vader_neu"] = scores.apply(lambda d: d.get("neu", 0))
    df["blob_polarity"]    = txt.apply(lambda s: TextBlob(s).polarity if pd.notna(s) else 0)
    df["blob_subjectivity"]= txt.apply(lambda s: TextBlob(s).subjectivity if pd.notna(s) else 0)

def tfidf_features(df, txt):
    vec = TfidfVectorizer(max_features=300, stop_words='english')
    X = vec.fit_transform(txt.fillna(""))
    df["tfidf_mean"] = np.mean(X.toarray(), axis=1)

def topic_features(df, txt):
    vec = TfidfVectorizer(max_features=300, stop_words='english')
    nmf = NMF(n_components=5, random_state=42)
    X = vec.fit_transform(txt.fillna(""))
    probs = nmf.fit_transform(X)
    for i in range(5):
        df[f"topic{i+1}_prob"] = probs[:, i]

def embedding_features(df, txt):
    print("Loading DistilBERT (CPU)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").eval()
    device = torch.device("cpu")
    model.to(device)

    embs = []
    for i in range(0, len(txt), BATCH_SIZE):
        batch = txt[i:i+BATCH_SIZE].fillna("").tolist()
        enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            cls = model(**enc).last_hidden_state[:, 0, :].cpu().numpy()
        embs.append(cls)
        del enc, cls
        gc.collect()
    emb_all = np.vstack(embs)
    df["distilbert_embedding"] = [row.tolist() for row in emb_all]
    print(f"Generated {len(emb_all)} embeddings", flush=True)

# ------------------- RUN ALL FEATURES -------------------
print("Starting feature extraction...", flush=True)
formatting_features(df, txt)
length_features(df, txt)
lexical_features(df, txt)
readability_features(df, txt)
sentiment_features(df, txt)
tfidf_features(df, txt)
topic_features(df, txt)
embedding_features(df, txt)

# ------------------- SAVE OUTPUT -------------------
output_path = os.path.join(OUTPUT_DIR, "features_output.parquet")
df.to_parquet(output_path, index=False, compression="snappy")
print(f"SUCCESS: Saved {len(df):,} rows to {output_path}", flush=True)
print("Job completed. Output in features_v2/", flush=True)