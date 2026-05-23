# ============================================================
# TextSense — Inference & Classification Report Script
# Loads all three saved models and runs evaluation on the
# held-out test set, then allows live custom text inference.
#
# USAGE:
#   python TextSense_Inference.py
#
# FOLDER STRUCTURE (all in the same folder):
#   TextSense_Inference.py
#   goemotions(FINAL).csv
#   textsense_outputs/
#       DistilBERT_best.pt
#       Bi-LSTM_best.pt
#       RoBERTa_best.pt
# ============================================================

import os
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast, DistilBertModel,
    RobertaTokenizerFast,    RobertaModel,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import (
    classification_report, f1_score,
    accuracy_score, precision_score,
    recall_score, confusion_matrix
)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATASET_PATH = 'goemotions(FINAL).csv'
OUTPUT_DIR   = 'textsense_outputs'

EMOTION_COLS = ['anger', 'sadness', 'joy', 'neutral']
NUM_LABELS   = len(EMOTION_COLS)

# ── Must match training script exactly ──
MAX_LEN      = 128
LSTM_MAX_LEN = 48
DROPOUT      = 0.2
EMBED_DIM    = 200
LSTM_HIDDEN  = 384
LSTM_LAYERS  = 2
LSTM_DROPOUT = 0.35
LSTM_VOCAB   = 20000
RANDOM_SEED  = 42

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()

print(f"{'='*55}")
print(f"  TextSense — Inference & Evaluation")
print(f"{'='*55}")
print(f"  Device : {device}")
if torch.cuda.is_available():
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
print(f"{'='*55}\n")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# DATASET PREPROCESSING
# Exact same pipeline as training script
# ─────────────────────────────────────────────

ANGER_EMOTIONS   = {'anger', 'annoyance', 'disgust', 'disapproval'}
SADNESS_EMOTIONS = {'sadness', 'grief', 'disappointment', 'remorse',
                    'embarrassment', 'nervousness', 'fear'}
JOY_EMOTIONS     = {'joy', 'love', 'gratitude', 'admiration', 'amusement',
                    'excitement', 'optimism', 'pride', 'relief', 'caring', 'desire'}
NEUTRAL_EMOTIONS = {'neutral', 'confusion', 'curiosity', 'realization',
                    'surprise', 'approval'}

GOE_EMOTION_COLS = [
    'admiration','amusement','anger','annoyance','approval','caring','confusion',
    'curiosity','desire','disappointment','disapproval','disgust','embarrassment',
    'excitement','fear','gratitude','grief','joy','love','nervousness','optimism',
    'pride','realization','relief','remorse','sadness','surprise','neutral'
]

GROUP_MAP = {}
for e in ANGER_EMOTIONS:   GROUP_MAP[e] = 'anger'
for e in SADNESS_EMOTIONS: GROUP_MAP[e] = 'sadness'
for e in JOY_EMOTIONS:     GROUP_MAP[e] = 'joy'
for e in NEUTRAL_EMOTIONS: GROUP_MAP[e] = 'neutral'

CONF_THRESHOLD = 0.40


def load_and_preprocess():
    # ── Load saved test split if available (guarantees same samples as training) ──
    test_split_path = os.path.join(OUTPUT_DIR, 'test_split.csv')

    if os.path.exists(test_split_path):
        print("Found saved test split — loading directly...")
        test_df = pd.read_csv(test_split_path)
        print(f"  Test samples: {len(test_df):,}")

        # Still need train_df to rebuild the LSTM vocab
        # Rebuild full dataset just for vocab purposes
        print("  Rebuilding dataset for LSTM vocab...")
        train_df = _rebuild_train_only()
        return train_df, None, test_df

    else:
        print("No saved test split found — rebuilding from scratch...")
        print("(Run training script first to save test_split.csv for reproducibility)")
        return _rebuild_full()


def _rebuild_train_only():
    """Rebuild only the training portion for LSTM vocab construction."""
    df_raw = pd.read_csv(DATASET_PATH)
    df_raw = df_raw[df_raw['example_very_unclear'] == False].copy()

    agg = df_raw.groupby('text')[GOE_EMOTION_COLS].sum().reset_index()

    def map_agg_to_4class(row):
        gs = {'anger': 0, 'sadness': 0, 'joy': 0, 'neutral': 0}
        for e in GOE_EMOTION_COLS:
            g = GROUP_MAP.get(e)
            if g:
                gs[g] += row[e]
        total = sum(gs.values())
        if total == 0:
            return None, 0.0
        best = max(gs, key=gs.get)
        return best, gs[best] / total

    results           = agg.apply(map_agg_to_4class, axis=1)
    agg['label']      = [r[0] for r in results]
    agg['confidence'] = [r[1] for r in results]

    df = agg[
        (agg['label'].notna()) & (agg['confidence'] >= CONF_THRESHOLD)
    ][['text', 'label']].copy()

    def clean_text(t):
        t = str(t).lower().strip()
        t = re.sub(r'\s+', ' ', t)
        return t

    df['text'] = df['text'].map(clean_text)
    df = df[df['text'].str.split().str.len() >= 3].reset_index(drop=True)

    target_per_class = int(df['label'].value_counts().max() * 0.60)
    balanced_parts   = []
    for lbl in df['label'].unique():
        cls_df = df[df['label'] == lbl]
        if len(cls_df) < target_per_class:
            cls_resampled = resample(cls_df, replace=True,
                                     n_samples=target_per_class,
                                     random_state=RANDOM_SEED)
        else:
            cls_resampled = resample(cls_df, replace=False,
                                     n_samples=target_per_class,
                                     random_state=RANDOM_SEED)
        balanced_parts.append(cls_resampled)
    df = pd.concat(balanced_parts).sample(
        frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    label2idx  = {e: i for i, e in enumerate(EMOTION_COLS)}
    df['label'] = df['label'].map(label2idx)

    train_df, _ = train_test_split(
        df, test_size=0.20, random_state=RANDOM_SEED, stratify=df['label'])
    return train_df


def _rebuild_full():
    """Full rebuild when no saved split exists — fallback only."""
    df_raw = pd.read_csv(DATASET_PATH)
    df_raw = df_raw[df_raw['example_very_unclear'] == False].copy()

    agg = df_raw.groupby('text')[GOE_EMOTION_COLS].sum().reset_index()

    def map_agg_to_4class(row):
        gs = {'anger': 0, 'sadness': 0, 'joy': 0, 'neutral': 0}
        for e in GOE_EMOTION_COLS:
            g = GROUP_MAP.get(e)
            if g:
                gs[g] += row[e]
        total = sum(gs.values())
        if total == 0:
            return None, 0.0
        best = max(gs, key=gs.get)
        return best, gs[best] / total

    results           = agg.apply(map_agg_to_4class, axis=1)
    agg['label']      = [r[0] for r in results]
    agg['confidence'] = [r[1] for r in results]

    df = agg[
        (agg['label'].notna()) & (agg['confidence'] >= CONF_THRESHOLD)
    ][['text', 'label']].copy()

    def clean_text(t):
        t = str(t).lower().strip()
        t = re.sub(r'\s+', ' ', t)
        return t

    df['text'] = df['text'].map(clean_text)
    df = df[df['text'].str.split().str.len() >= 3].reset_index(drop=True)

    target_per_class = int(df['label'].value_counts().max() * 0.60)
    balanced_parts   = []
    for lbl in df['label'].unique():
        cls_df = df[df['label'] == lbl]
        if len(cls_df) < target_per_class:
            cls_resampled = resample(cls_df, replace=True,
                                     n_samples=target_per_class,
                                     random_state=RANDOM_SEED)
        else:
            cls_resampled = resample(cls_df, replace=False,
                                     n_samples=target_per_class,
                                     random_state=RANDOM_SEED)
        balanced_parts.append(cls_resampled)
    df = pd.concat(balanced_parts).sample(
        frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    label2idx  = {e: i for i, e in enumerate(EMOTION_COLS)}
    df['label'] = df['label'].map(label2idx)

    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=RANDOM_SEED, stratify=df['label'])
    val_df, test_df   = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_SEED,
        stratify=temp_df['label'])

    print(f"  Train : {len(train_df):,}")
    print(f"  Val   : {len(val_df):,}")
    print(f"  Test  : {len(test_df):,}")
    return train_df, val_df, test_df


# ─────────────────────────────────────────────
# DATASET CLASSES
# ─────────────────────────────────────────────

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts.reset_index(drop=True)
        self.labels    = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }


class LSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts   = texts.reset_index(drop=True)
        self.labels  = labels.reset_index(drop=True)
        self.vocab   = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = _tokenize(self.texts[idx])[:self.max_len]
        ids    = [self.vocab.get(t, 1) for t in tokens]
        ids    = ids + [0] * (self.max_len - len(ids))
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'labels':    torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }


# ─────────────────────────────────────────────
# VOCAB HELPERS
# ─────────────────────────────────────────────

def _tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def build_vocab(texts, max_vocab=20000):
    counter = Counter()
    for t in texts:
        counter.update(_tokenize(t))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# Must match training script exactly
# ─────────────────────────────────────────────

class DistilBERTClassifier(nn.Module):
    def __init__(self, num_labels, dropout=DROPOUT):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        hidden          = self.distilbert.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask):
        out  = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls  = self.dropout(out.last_hidden_state[:, 0, :])
        proj = self.projection(cls)
        return self.classifier(proj)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers, num_labels, dropout,
                 pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)
        self.bilstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True, bidirectional=True
        )
        self.dropout    = nn.Dropout(dropout)
        self.attn_fc    = nn.Linear(hidden_dim * 2, 1)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        self.classifier = nn.Linear(128, num_labels)

    def forward(self, input_ids, attention_mask=None):
        x, _    = self.bilstm(self.embedding(input_ids))
        mask    = (input_ids != 0)
        scores  = self.attn_fc(x).squeeze(-1)
        scores  = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (weights * x).sum(dim=1)
        context = self.layer_norm(context)
        proj    = self.projection(self.dropout(context))
        return self.classifier(proj)


class RoBERTaClassifier(nn.Module):
    def __init__(self, num_labels, dropout=DROPOUT):
        super().__init__()
        self.roberta    = RobertaModel.from_pretrained('roberta-base')
        hidden          = self.roberta.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask):
        out  = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls  = self.dropout(out.last_hidden_state[:, 0, :])
        proj = self.projection(cls)
        return self.classifier(proj)


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def get_predictions(model, test_df, tokenizer=None,
                    vocab=None, is_lstm=False, batch_size=32):
    if is_lstm:
        dataset = LSTMDataset(test_df['text'], test_df['label'],
                              vocab, LSTM_MAX_LEN)
    else:
        dataset = EmotionDataset(test_df['text'], test_df['label'],
                                 tokenizer, MAX_LEN)

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            lbl = batch['labels'].numpy()
            if is_lstm:
                logits = model(ids)
            else:
                mask   = batch['attention_mask'].to(device)
                logits = model(ids, mask)
            all_preds.extend(
                torch.softmax(logits.float(), dim=1).cpu().numpy())
            all_labels.extend(lbl)

    return np.array(all_labels), np.array(all_preds)


def print_classification_report(labels, preds, model_name):
    preds_bin   = np.argmax(preds, axis=1)
    accuracy    = accuracy_score(labels, preds_bin)
    macro_f1    = f1_score(labels, preds_bin, average='macro',    zero_division=0)
    weighted_f1 = f1_score(labels, preds_bin, average='weighted', zero_division=0)
    precision   = precision_score(labels, preds_bin, average='macro', zero_division=0)
    recall      = recall_score(labels, preds_bin, average='macro',    zero_division=0)

    print(f"\n{'='*65}")
    print(f"  {model_name} — Classification Report")
    print(f"{'='*65}")
    print(f"  Accuracy        : {accuracy:.4f}  "
          f"{'✓ ≥80%' if accuracy >= 0.80 else '✗ <80%'}")
    print(f"  Macro Precision : {precision:.4f}")
    print(f"  Macro Recall    : {recall:.4f}")
    print(f"  Macro F1        : {macro_f1:.4f}")
    print(f"  Weighted F1     : {weighted_f1:.4f}")
    print(f"{'='*65}")
    print(classification_report(
        labels, preds_bin,
        target_names=EMOTION_COLS, zero_division=0))

    return dict(
        model=model_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
    )


def plot_confusion_matrix(labels, preds, model_name):
    preds_bin = np.argmax(preds, axis=1)
    cm = confusion_matrix(labels, preds_bin)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_COLS,
                yticklabels=EMOTION_COLS)
    acc = accuracy_score(labels, preds_bin)
    plt.title(f'{model_name} — Confusion Matrix\nAccuracy: {acc:.4f}',
              fontsize=13, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    fname = os.path.join(
        OUTPUT_DIR,
        f"{model_name.lower().replace(' ', '_')}_inference_confusion.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_comparison(all_results):
    models = [r['model'] for r in all_results]
    colors = ['#4C72B0', '#55A868', '#C44E52']
    metrics_to_plot = [
        ('accuracy',    'Accuracy'),
        ('precision',   'Macro Precision'),
        ('recall',      'Macro Recall'),
        ('macro_f1',    'Macro F1'),
        ('weighted_f1', 'Weighted F1'),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 6))
    fig.suptitle('TextSense — Model Comparison (Test Set, 4-Class)',
                 fontsize=14, fontweight='bold')

    for ax, (key, title) in zip(axes, metrics_to_plot):
        vals = [r[key] for r in all_results]
        bars = ax.bar(models, vals, color=colors)
        ax.axhline(0.80, color='green', linestyle='--',
                   alpha=0.7, label='80% target')
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_ylim(0, min(max(vals) * 1.20, 1.05))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f'{v:.4f}', ha='center',
                    fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'inference_comparison.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────

def predict_all(text, distilbert_model, bilstm_model, roberta_model,
                db_tokenizer, rb_tokenizer, vocab):
    print(f'\n{"="*60}')
    print(f'  Text: "{text}"')
    print(f'{"="*60}')

    models_list = [
        ('DistilBERT', distilbert_model, db_tokenizer, False),
        ('Bi-LSTM',    bilstm_model,     None,         True),
        ('RoBERTa',    roberta_model,    rb_tokenizer, False),
    ]

    for model_name, model, tokenizer, is_lstm in models_list:
        model.eval()
        with torch.no_grad():
            if is_lstm:
                tokens = _tokenize(text)[:LSTM_MAX_LEN]
                ids    = [vocab.get(t, 1) for t in tokens]
                ids    = ids + [0] * (LSTM_MAX_LEN - len(ids))
                tensor = torch.tensor([ids], dtype=torch.long).to(device)
                logits = model(tensor)
            else:
                enc    = tokenizer(
                    text, truncation=True, padding='max_length',
                    max_length=MAX_LEN, return_tensors='pt')
                logits = model(
                    enc['input_ids'].to(device),
                    enc['attention_mask'].to(device))
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]

        predicted = EMOTION_COLS[np.argmax(probs)]
        print(f'\n  [{model_name}] → {predicted.upper()}')
        for emotion, score in sorted(zip(EMOTION_COLS, probs),
                                     key=lambda x: x[1], reverse=True):
            bar = '█' * int(score * 25)
            print(f'    {emotion:<12} {score:.4f}  {bar}')


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    # ── 1. Load and preprocess dataset ──
    train_df, val_df, test_df = load_and_preprocess()

    # ── 2. Build LSTM vocab from training split ──
    print("\nBuilding vocabulary...")
    vocab = build_vocab(train_df['text'].tolist(), max_vocab=LSTM_VOCAB)
    print(f"  Vocabulary size: {len(vocab):,}")

    # ── Confirm test split source ──
    test_split_path = os.path.join(OUTPUT_DIR, 'test_split.csv')
    if os.path.exists(test_split_path):
        print(f"  ✓ Using saved test split ({len(test_df):,} samples) — results reproducible")
    else:
        print(f"  ⚠ Using rebuilt test split — run training script to fix reproducibility")

    # ── 3. Load tokenizers ──
    print("\nLoading tokenizers...")
    db_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    rb_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    print("  ✓ Tokenizers ready")

    # ── 4. Load models ──
    print("\nLoading model weights...")

    distilbert_model = DistilBERTClassifier(NUM_LABELS).to(device)
    distilbert_model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, 'DistilBERT_best.pt'),
                   map_location=device))
    distilbert_model.eval()
    print("  ✓ DistilBERT loaded")

    bilstm_model = BiLSTMClassifier(
        vocab_size=LSTM_VOCAB, embed_dim=EMBED_DIM,
        hidden_dim=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
        num_labels=NUM_LABELS, dropout=LSTM_DROPOUT
    ).to(device)
    bilstm_model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, 'Bi-LSTM_best.pt'),
                   map_location=device))
    bilstm_model.eval()
    print("  ✓ Bi-LSTM loaded")

    roberta_model = RoBERTaClassifier(NUM_LABELS).to(device)
    roberta_model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, 'RoBERTa_best.pt'),
                   map_location=device))
    roberta_model.eval()
    print("  ✓ RoBERTa loaded")

    # ── 5. Run evaluation on test set ──
    print("\nRunning evaluation on test set...")
    labels_db,   preds_db   = get_predictions(
        distilbert_model, test_df, tokenizer=db_tokenizer)
    labels_lstm, preds_lstm = get_predictions(
        bilstm_model, test_df, vocab=vocab, is_lstm=True)
    labels_rb,   preds_rb   = get_predictions(
        roberta_model, test_df, tokenizer=rb_tokenizer)
    print("  ✓ Predictions complete")

    # ── 6. Print classification reports ──
    results_db   = print_classification_report(labels_db,   preds_db,   'DistilBERT')
    results_lstm = print_classification_report(labels_lstm, preds_lstm, 'Bi-LSTM')
    results_rb   = print_classification_report(labels_rb,   preds_rb,   'RoBERTa')

    # ── 7. Confusion matrices ──
    print("\nGenerating confusion matrices...")
    plot_confusion_matrix(labels_db,   preds_db,   'DistilBERT')
    plot_confusion_matrix(labels_lstm, preds_lstm, 'Bi-LSTM')
    plot_confusion_matrix(labels_rb,   preds_rb,   'RoBERTa')

    # ── 8. Summary table ──
    all_results = [results_db, results_lstm, results_rb]
    print(f"\n{'='*70}")
    print(f"  TEXTSENSE — FINAL COMPARATIVE SUMMARY")
    print(f"{'='*70}")
    summary = pd.DataFrame(all_results)
    summary['≥80%?'] = summary['accuracy'].apply(
        lambda x: '✓' if x >= 0.80 else '✗')
    summary = summary.rename(columns={
        'model':       'Model',
        'accuracy':    'Accuracy',
        'precision':   'Precision (M)',
        'recall':      'Recall (M)',
        'macro_f1':    'Macro F1',
        'weighted_f1': 'Weighted F1',
    })
    print(summary.to_string(index=False))

    # ── 9. Comparison bar chart ──
    plot_comparison(all_results)

    # ── 10. Live inference demo ──
    print(f"\n{'='*60}")
    print("  LIVE INFERENCE DEMO")
    print(f"{'='*60}")

    demo_texts = [
        "This product is absolutely amazing, best purchase I've ever made!",
        "Terrible quality, broke after one use. Complete waste of money.",
        "It works fine, does exactly what it says. Nothing special.",
        "I am so frustrated, this never works and support is useless.",
        "I cried when it arrived, the packaging was so thoughtful.",
    ]

    for text in demo_texts:
        predict_all(text, distilbert_model, bilstm_model, roberta_model,
                    db_tokenizer, rb_tokenizer, vocab)
     # ── 11. Custom input loop ──
    print(f"\n{'='*60}")
    print("  CUSTOM INPUT — type your own text")
    print("  (type 'quit' to exit)")
    print(f"{'='*60}")
 
    while True:
        try:
            user_input = input("\nEnter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ('quit', 'exit', 'q', ''):
            break
        predict_all(user_input, distilbert_model, bilstm_model, roberta_model,
                    db_tokenizer, rb_tokenizer, vocab)
 
    print("\n  Done. Plots saved to:", OUTPUT_DIR)

if __name__ == '__main__':
    main()