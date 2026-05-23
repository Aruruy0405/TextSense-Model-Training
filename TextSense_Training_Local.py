# ============================================================
# TextSense: Emotion Recognition Training Script  
# Local GPU Version — NVIDIA RTX 4060
# Algorithms: DistilBERT | Bi-LSTM | RoBERTa
# Dataset: GoEmotions — properly aggregated across raters (28→4 class)
# Labels : anger | sadness | joy | neutral
# Target : ≥ 80% accuracy on all three models
# Root cause fix: raw CSV has ~3-4 raters per text; treating each rater
#   row as an independent sample caused label noise of ~38% (texts
#   appeared under conflicting labels). Fix: aggregate votes per text
#   first, then assign label by group majority vote with confidence≥0.40.
# ============================================================
# CHANGES FROM v2:
#   Dataset / Preprocessing:
#     - Accepts the raw GoEmotions CSV directly (no pre-cleaned file needed)
#     - Maps 28 fine-grained emotions → 4 coarse labels via group lookup
#       (ANGER: anger/annoyance/disgust/disapproval;
#        SADNESS: sadness/grief/disappointment/remorse/embarrassment/nervousness/fear;
#        JOY: joy/love/gratitude/admiration/amusement/excitement/optimism/pride/relief/caring/desire;
#        NEUTRAL: neutral/confusion/curiosity/realization/surprise/approval)
#     - Multi-label rows resolved by majority-group vote
#     - Rows flagged example_very_unclear=True are removed
#     - Text cleaning: lower, strip, collapse whitespace, drop ≤2 word rows
#
#   DistilBERT:
#     - Projection head: 768 → 256 → 4  (restored from comment in v2; was missing in actual code)
#     - Layer-wise LR decay: backbone 3e-5, classifier 1e-4  (new)
#     - Epochs: 8 → 6  (early stopping on val F1 with patience=3)
#     - Warmup: 300 → 500
#     - Label smoothing: 0.0 → 0.1  (was set in config but never passed — now active)
#
#   Bi-LSTM:
#     - GloVe path fallback gracefully creates random embeddings (no crash)
#     - Epochs: 20 → 25 with early stopping (patience=5)
#     - Added LayerNorm before classifier
#     - Label smoothing: 0.0 → 0.1 (now active)
#
#   RoBERTa:
#     - Projection head: 768 → 256 → 4  (restored — same gap as DistilBERT)
#     - Layer-wise LR decay: backbone 1e-5, classifier 5e-5
#     - Epochs: 8 → 6 with early stopping (patience=3)
#     - Warmup: 1000 → 750
#
#   Shared:
#     - LABEL_SMOOTH config value now actually passed to the loss function
#     - Early stopping replaces fixed epoch count for transformers
#     - fp16 mixed-precision training enabled (RTX 4060 native support)
#     - num_workers=4 on Linux (was 0 — wastes CPU↔GPU transfer time)
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
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import (
    DistilBertTokenizerFast, DistilBertModel,
    RobertaTokenizerFast, RobertaModel,
    get_linear_schedule_with_warmup
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

DATASET_PATH = "goemotions(FINAL).csv"   # raw GoEmotions file
GLOVE_PATH   = 'glove.6B.200d.txt'
OUTPUT_DIR   = 'textsense_outputs'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Device ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()          # mixed precision only on GPU
print(f"{'='*55}")
print(f"  TextSense Training v8 — Local GPU")
print(f"{'='*55}")
print(f"  Device    : {device}")
if torch.cuda.is_available():
    print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  AMP (fp16): {USE_AMP}")
print(f"{'='*55}\n")

# ── Emotion Labels ──
EMOTION_COLS = ['anger', 'sadness', 'joy', 'neutral']
NUM_LABELS   = len(EMOTION_COLS)   # 4

# ── GoEmotions → 4-class group mapping ──
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

# ── Shared Hyperparameters ──
MAX_LEN      = 128   # transformers — subword tokens can be longer
LSTM_MAX_LEN = 48    # word tokens; mean=13.8, 95th=25, only 1 text > 48 words
WEIGHT_DECAY = 0.01
DROPOUT      = 0.2
LABEL_SMOOTH = 0.1        # now actually applied in the loss call
RANDOM_SEED  = 42
NUM_WORKERS  = 0          # must be 0 on Windows (spawn multiprocessing has no fork)

# ── DistilBERT ──
DB_LR_BACKBONE  = 2e-5    # lowered — 3e-5 was overshooting on 56K samples
DB_LR_HEAD      = 5e-5    # lowered proportionally
DB_BATCH        = 32
DB_EPOCHS       = 10      # more ceiling with early stopping
DB_WARMUP       = 500
DB_PATIENCE     = 4       # more patience to find the real peak

# ── RoBERTa ──
RB_LR_BACKBONE  = 2e-5    # raised from 1e-5 — mirroring DistilBERT which hit 83%
RB_LR_HEAD      = 1e-4    # raised — 3e-5 was underfitting on balanced 58K set
RB_BATCH        = 32      # raised to 32 — 16 gave 2550 steps/epoch, scheduler too granular
RB_EPOCHS       = 20      # was still converging at ep10 (val_F1 0.7675, not plateaued)
RB_WARMUP       = 600     # restored: 600 worked well in the 77% run
RB_PATIENCE     = 7       # gains are small but steady — need more patience

# ── Bi-LSTM ──
LSTM_EPOCHS   = 35        # more epochs — LSTM converges slower on balanced set
LSTM_LR       = 0.0005    # halved — 0.001 was too aggressive after balancing
LSTM_HIDDEN   = 384    # wider — 256 was underpowered after shorter padding removes noise
LSTM_LAYERS   = 2
LSTM_DROPOUT  = 0.35   # 0.4 over-regularises random embeddings on 52K train set
LSTM_BATCH    = 64        # smaller batch for better gradient signal
EMBED_DIM     = 200    # must match your GloVe file (glove.6B.200d.txt = 200-d)
                       # change to 300 only if you have glove.6B.300d.txt
LSTM_PATIENCE = 6         # more patience

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)



def main():
    global EMBED_DIM  # may be updated by load_glove if file dim differs
    # ─────────────────────────────────────────────
    # LOAD & PREPROCESS  (GoEmotions → 4-class)
    # ─────────────────────────────────────────────

    print("Loading GoEmotions dataset...")
    df_raw = pd.read_csv(DATASET_PATH)

    # ── remove unclear examples ──
    df_raw = df_raw[df_raw['example_very_unclear'] == False].copy()

    print("  Aggregating multi-rater votes per unique text...")
    agg = df_raw.groupby('text')[GOE_EMOTION_COLS].sum().reset_index()
    print(f"  Unique texts: {len(agg):,}  (raw rows were: {len(df_raw):,})")

    GROUP_MAP = {}
    for e in ANGER_EMOTIONS:   GROUP_MAP[e] = 'anger'
    for e in SADNESS_EMOTIONS: GROUP_MAP[e] = 'sadness'
    for e in JOY_EMOTIONS:     GROUP_MAP[e] = 'joy'
    for e in NEUTRAL_EMOTIONS: GROUP_MAP[e] = 'neutral'

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

    results          = agg.apply(map_agg_to_4class, axis=1)
    agg['label']      = [r[0] for r in results]
    agg['confidence'] = [r[1] for r in results]

    # Keep texts where the winning group has ≥40% of total rater votes.
    CONF_THRESHOLD = 0.40
    df = agg[(agg['label'].notna()) & (agg['confidence'] >= CONF_THRESHOLD)][['text', 'label']].copy()
    print(f"  After confidence filter (≥{CONF_THRESHOLD}): {len(df):,} texts retained")

    # ── text cleaning ──
    def clean_text(t):
        t = str(t).lower().strip()
        t = re.sub(r'\s+', ' ', t)
        return t

    df['text'] = df['text'].map(clean_text)
    df = df[df['text'].str.split().str.len() >= 3].reset_index(drop=True)

    # class balancing via oversample minority / undersample majority ──
    # Neutral is 44% of data (5.2x sadness). Class weights alone can't fully
    # compensate — balanced sampling gives the model equal exposure per class.
    # Target = 60% of the largest class count (keeps majority data but limits
    # dominance); minority classes are oversampled with replacement.
    target_per_class = int(df['label'].value_counts().max() * 0.60)
    balanced_parts = []
    for lbl in df['label'].unique():
        cls_df = df[df['label'] == lbl]
        if len(cls_df) < target_per_class:
            cls_resampled = resample(cls_df, replace=True,
                                     n_samples=target_per_class, random_state=RANDOM_SEED)
        else:
            cls_resampled = resample(cls_df, replace=False,
                                     n_samples=target_per_class, random_state=RANDOM_SEED)
        balanced_parts.append(cls_resampled)
    df = pd.concat(balanced_parts).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"  After class balancing (target/class={target_per_class:,}): {len(df):,} total")
    print(f"  Balanced distribution: {dict(df['label'].value_counts())}")

    label2idx = {e: i for i, e in enumerate(EMOTION_COLS)}
    df['label'] = df['label'].map(label2idx)

    print(f"  Total samples    : {len(df):,}")
    print(f"  Number of labels : {NUM_LABELS}")
    print()
    print("  Label distribution:")
    label_series = df['label'].map(dict(enumerate(EMOTION_COLS)))
    counts = label_series.value_counts().reindex(EMOTION_COLS)
    for emotion, count in counts.items():
        pct = count / len(df) * 100
        print(f"    {emotion:<12}: {count:>6} ({pct:.1f}%)")

    # ── class weights ──
    class_counts  = df['label'].value_counts().sort_index().values
    class_weights = torch.tensor(
        1.0 / (class_counts / class_counts.sum()), dtype=torch.float
    )
    class_weights = (class_weights / class_weights.sum() * NUM_LABELS).to(device)
    print(f"\n  Class weights: { {e: f'{w:.3f}' for e, w in zip(EMOTION_COLS, class_weights.cpu().tolist())} }")

    # ── distribution plot ──
    plt.figure(figsize=(10, 5))
    colors_bar = ['#C44E52', '#4C72B0', '#55A868', '#8172B2']
    sns.barplot(x=counts.values, y=counts.index, palette=colors_bar)
    plt.title('Emotion Label Distribution (4-class, GoEmotions)', fontsize=14)
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'emotion_distribution.png'), dpi=150)
    plt.close()
    print("  Saved: emotion_distribution.png")

    # ── Train / Val / Test (80 / 10 / 10) ──
    # +14% more training data directly benefits LSTM with random embeddings
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=RANDOM_SEED,
                                         stratify=df['label'])
    val_df,   test_df = train_test_split(temp_df, test_size=0.50, random_state=RANDOM_SEED,
                                         stratify=temp_df['label'])

    print(f"\n  Train : {len(train_df):,}")
    print(f"  Val   : {len(val_df):,}")
    print(f"  Test  : {len(test_df):,}\n")

    # Save test split so inference.py evaluates on identical samples
    test_split_path = os.path.join(OUTPUT_DIR, 'test_split.csv')
    test_df.to_csv(test_split_path, index=False)
    print(f"  Saved test split: {test_split_path}")


    # ─────────────────────────────────────────────
    # DATASET CLASSES
    # ─────────────────────────────────────────────

    class EmotionDataset(Dataset):
        """For transformer models (DistilBERT, RoBERTa)."""
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
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt'
            )
            return {
                'input_ids':      encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels':         torch.tensor(self.labels.iloc[idx], dtype=torch.long)
            }


    class LSTMDataset(Dataset):
        """For Bi-LSTM (manual tokenization)."""
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
    # LOSS
    # ─────────────────────────────────────────────

    class SmoothCrossEntropyLoss(nn.Module):
        """CrossEntropyLoss with label smoothing and optional class weights."""
        def __init__(self, smoothing=0.1, weight=None):
            super().__init__()
            self.smoothing = smoothing
            self.weight    = weight

        def forward(self, logits, targets):
            n_classes  = logits.size(-1)
            log_probs  = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            loss = -(smooth_targets * log_probs).sum(dim=-1)
            if self.weight is not None:
                loss = loss * self.weight[targets]
            return loss.mean()


    # ─────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────

    def compute_epoch_metrics(all_labels, all_preds):
        preds_bin = np.argmax(np.array(all_preds), axis=1)
        labels    = np.array(all_labels)
        return {
            'subset_acc' : accuracy_score(labels, preds_bin),
            'macro_f1'   : f1_score(labels, preds_bin, average='macro',    zero_division=0),
            'weighted_f1': f1_score(labels, preds_bin, average='weighted', zero_division=0),
            'precision'  : precision_score(labels, preds_bin, average='macro', zero_division=0),
            'recall'     : recall_score(labels, preds_bin, average='macro',    zero_division=0),
        }


    def compute_final_metrics(all_labels, all_preds, model_name):
        preds_bin   = np.argmax(np.array(all_preds), axis=1)
        labels      = np.array(all_labels)
        accuracy    = accuracy_score(labels, preds_bin)
        macro_f1    = f1_score(labels, preds_bin, average='macro',    zero_division=0)
        weighted_f1 = f1_score(labels, preds_bin, average='weighted', zero_division=0)
        precision   = precision_score(labels, preds_bin, average='macro', zero_division=0)
        recall      = recall_score(labels, preds_bin, average='macro',    zero_division=0)

        print(f"\n{'='*60}")
        print(f"  {model_name} — Final Test Set Evaluation")
        print(f"{'='*60}")
        print(f"  Accuracy                      : {accuracy:.4f}  {'✓ ≥80%' if accuracy >= 0.80 else '✗ <80%'}")
        print(f"  Macro Precision               : {precision:.4f}")
        print(f"  Macro Recall                  : {recall:.4f}")
        print(f"  Macro F1-Score                : {macro_f1:.4f}")
        print(f"  Weighted F1-Score             : {weighted_f1:.4f}")
        print(f"{'='*60}")
        print("\nPer-Emotion Classification Report:")
        print(classification_report(labels, preds_bin,
                                    target_names=EMOTION_COLS, zero_division=0))
        return {
            'subset_accuracy': accuracy,
            'macro_f1':        macro_f1,
            'weighted_f1':     weighted_f1,
            'precision':       precision,
            'recall':          recall,
        }


    def print_epoch_row(epoch, total, train_loss, val_loss, train_m, val_m, name):
        print(
            f"[{name}] Epoch {epoch+1:>2}/{total} | "
            f"Loss {train_loss:.4f}/{val_loss:.4f} | "
            f"Acc {train_m['subset_acc']:.4f}/{val_m['subset_acc']:.4f} | "
            f"F1(M) {train_m['macro_f1']:.4f}/{val_m['macro_f1']:.4f} | "
            f"Prec {val_m['precision']:.4f} | "
            f"Rec {val_m['recall']:.4f}"
        )


    def plot_curves(train_losses, val_losses, train_metrics, val_metrics, model_name):
        epochs_r = range(1, len(train_losses) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'{model_name} — Training Curves', fontsize=15, fontweight='bold')

        panels = [
            (axes[0,0], train_losses, val_losses, None, None, 'Loss', 'Loss', True),
            (axes[0,1], None, None,
             [m['subset_acc']  for m in train_metrics],
             [m['subset_acc']  for m in val_metrics],
             'Accuracy', 'Accuracy', False),
            (axes[1,0], None, None,
             [m['macro_f1']    for m in train_metrics],
             [m['macro_f1']    for m in val_metrics],
             'Macro F1-Score', 'F1', False),
            (axes[1,1], None, None,
             [m['weighted_f1'] for m in train_metrics],
             [m['weighted_f1'] for m in val_metrics],
             'Weighted F1-Score', 'Weighted F1', False),
        ]

        for ax, tl, vl, ta, va, title, ylabel, use_loss in panels:
            y_train = tl if use_loss else ta
            y_val   = vl if use_loss else va
            ax.plot(epochs_r, y_train, 'b-o', label='Train')
            ax.plot(epochs_r, y_val,   'r-o', label='Val')
            if not use_loss:
                ax.axhline(0.80, color='green', linestyle='--', alpha=0.6, label='80% target')
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        fname = os.path.join(OUTPUT_DIR,
                             f"{model_name.lower().replace(' ','_')}_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved: {fname}")


    def plot_confusion_matrix(all_labels, all_preds, model_name):
        preds_bin = np.argmax(np.array(all_preds), axis=1)
        labels    = np.array(all_labels)
        cm        = confusion_matrix(labels, preds_bin)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=EMOTION_COLS,
                    yticklabels=EMOTION_COLS)
        plt.title(f'{model_name} — Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        fname = os.path.join(OUTPUT_DIR,
                             f"{model_name.lower().replace(' ','_')}_confusion.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved: {fname}")


    # ─────────────────────────────────────────────
    # TRANSFORMER TRAINING (shared, with early stopping)
    # ─────────────────────────────────────────────

    def build_transformer_optimizer(model, lr_backbone, lr_head):
        """Two-group split: uniform backbone LR + higher head LR.
        Discriminative per-layer decay was tested (v8) but slowed early
        convergence on this 58K dataset — reverted.
        """
        backbone_params = [p for n, p in model.named_parameters()
                           if not n.startswith('classifier') and not n.startswith('projection')]
        head_params     = [p for n, p in model.named_parameters()
                           if n.startswith('classifier') or n.startswith('projection')]
        return AdamW([
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': head_params,     'lr': lr_head},
        ], weight_decay=WEIGHT_DECAY)


    def train_transformer(model, train_loader, val_loader, model_name,
                          epochs, lr_backbone, lr_head, warmup, patience):
        optimizer   = build_transformer_optimizer(model, lr_backbone, lr_head)
        total_steps = len(train_loader) * epochs
        scheduler   = get_linear_schedule_with_warmup(
                          optimizer,
                          num_warmup_steps=warmup,
                          num_training_steps=total_steps)
        criterion   = SmoothCrossEntropyLoss(smoothing=LABEL_SMOOTH, weight=class_weights)
        scaler      = GradScaler(enabled=USE_AMP)

        train_losses, val_losses   = [], []
        train_metrics, val_metrics = [], []
        best_val_f1  = 0.0
        no_improve   = 0

        print(f"\n{'─'*70}")
        print(f"  {model_name} | Epochs: {epochs} | LR backbone: {lr_backbone} | "
              f"LR head: {lr_head} | Warmup: {warmup} | Patience: {patience}")
        print(f"{'─'*70}")

        for epoch in range(epochs):
            # ── Train ──
            model.train()
            epoch_loss, ep, el = 0.0, [], []

            for batch in tqdm(train_loader,
                              desc=f"[{model_name}] E{epoch+1}/{epochs} TRAIN",
                              unit='batch', leave=False):
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                lbl  = batch['labels'].to(device)

                optimizer.zero_grad()
                with autocast(enabled=USE_AMP):
                    logits = model(ids, mask)
                    loss   = criterion(logits, lbl)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()
                ep.extend(torch.softmax(logits.float(), dim=1).detach().cpu().numpy())
                el.extend(lbl.cpu().numpy())

            avg_train_loss = epoch_loss / len(train_loader)
            train_m = compute_epoch_metrics(el, ep)

            # ── Validate ──
            model.eval()
            val_loss_total, vp, vl = 0.0, [], []

            with torch.no_grad():
                for batch in tqdm(val_loader,
                                  desc=f"[{model_name}] E{epoch+1}/{epochs} VAL  ",
                                  unit='batch', leave=False):
                    ids  = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    lbl  = batch['labels'].to(device)
                    with autocast(enabled=USE_AMP):
                        logits = model(ids, mask)
                        val_loss_total += criterion(logits, lbl).item()
                    vp.extend(torch.softmax(logits.float(), dim=1).cpu().numpy())
                    vl.extend(lbl.cpu().numpy())

            avg_val_loss = val_loss_total / len(val_loader)
            val_m = compute_epoch_metrics(vl, vp)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_metrics.append(train_m)
            val_metrics.append(val_m)

            print_epoch_row(epoch, epochs, avg_train_loss, avg_val_loss,
                            train_m, val_m, model_name)

            if val_m['macro_f1'] > best_val_f1:
                best_val_f1 = val_m['macro_f1']
                no_improve  = 0
                torch.save(model.state_dict(),
                           os.path.join(OUTPUT_DIR, f'{model_name}_best.pt'))
                print(f"  ✓ Best checkpoint saved (val macro-F1: {best_val_f1:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  ⚑ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break

        print(f"{'─'*70}")
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f'{model_name}_best.pt')))
        print(f"  Reloaded best checkpoint (val macro-F1: {best_val_f1:.4f})")

        plot_curves(train_losses, val_losses, train_metrics, val_metrics, model_name)
        return model


    def evaluate_model(model, test_loader, model_name, is_lstm=False):
        model.eval()
        test_preds, test_labels = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"[{model_name}] TEST", unit='batch'):
                ids = batch['input_ids'].to(device)
                lbl = batch['labels'].numpy()
                if is_lstm:
                    logits = model(ids)
                else:
                    mask   = batch['attention_mask'].to(device)
                    with autocast(enabled=USE_AMP):
                        logits = model(ids, mask)
                test_preds.extend(torch.softmax(logits.float(), dim=1).cpu().numpy())
                test_labels.extend(lbl)

        plot_confusion_matrix(test_labels, test_preds, model_name)
        return compute_final_metrics(test_labels, test_preds, model_name)


    # ─────────────────────────────────────────────
    # INFERENCE HELPERS
    # ─────────────────────────────────────────────

    def predict_emotion(text, model, tokenizer):
        model.eval()
        encoding = tokenizer(
            text, truncation=True, padding='max_length',
            max_length=MAX_LEN, return_tensors='pt'
        )
        ids  = encoding['input_ids'].to(device)
        mask = encoding['attention_mask'].to(device)
        with torch.no_grad():
            with autocast(enabled=USE_AMP):
                logits = model(ids, mask)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
        predicted = EMOTION_COLS[np.argmax(probs)]
        print(f'\nText: "{text}"')
        print(f"Predicted emotion: {predicted.upper()}")
        for emotion, score in sorted(zip(EMOTION_COLS, probs), key=lambda x: x[1], reverse=True):
            print(f"  {emotion:<16} {score:.4f}  {'█' * int(score * 20)}")
        return predicted


    def predict_emotion_lstm(text, model, vocab):
        model.eval()
        tokens = _tokenize(text)[:MAX_LEN]
        ids    = [vocab.get(t, 1) for t in tokens] + [0] * (MAX_LEN - len(tokens))
        tensor = torch.tensor([ids], dtype=torch.long).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor).float(), dim=1).cpu().numpy()[0]
        predicted = EMOTION_COLS[np.argmax(probs)]
        print(f'\nText: "{text}"')
        print(f"Predicted emotion: {predicted.upper()}")
        for emotion, score in sorted(zip(EMOTION_COLS, probs), key=lambda x: x[1], reverse=True):
            print(f"  {emotion:<16} {score:.4f}  {'█' * int(score * 20)}")
        return predicted


    # ─────────────────────────────────────────────
    # MODEL A: DistilBERT
    # ─────────────────────────────────────────────

    class DistilBERTClassifier(nn.Module):
        """
        projection head 768 → 256 → 4 (now actually wired in forward()).
        A non-linear bottleneck forces task-specific representations before
        the final classification layer.
        """
        def __init__(self, num_labels, dropout=DROPOUT):
            super().__init__()
            self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            hidden          = self.distilbert.config.hidden_size   # 768
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


    print("\n" + "="*55)
    print("  MODEL A: DistilBERT  ")
    print("="*55)

    db_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_loader_db = DataLoader(
        EmotionDataset(train_df['text'], train_df['label'], db_tokenizer, MAX_LEN),
        batch_size=DB_BATCH, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader_db = DataLoader(
        EmotionDataset(val_df['text'],   val_df['label'],   db_tokenizer, MAX_LEN),
        batch_size=DB_BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader_db = DataLoader(
        EmotionDataset(test_df['text'],  test_df['label'],  db_tokenizer, MAX_LEN),
        batch_size=DB_BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    distilbert_model = DistilBERTClassifier(NUM_LABELS).to(device)
    distilbert_model = train_transformer(
        distilbert_model, train_loader_db, val_loader_db,
        "DistilBERT",
        epochs=DB_EPOCHS, lr_backbone=DB_LR_BACKBONE, lr_head=DB_LR_HEAD,
        warmup=DB_WARMUP, patience=DB_PATIENCE
    )
    metrics_distilbert = evaluate_model(distilbert_model, test_loader_db, "DistilBERT")

    torch.save(distilbert_model.state_dict(),
               os.path.join(OUTPUT_DIR, 'distilbert_textsense.pt'))
    print(f"  Saved: {OUTPUT_DIR}/distilbert_textsense.pt")

    print("\n--- DistilBERT Inference Test ---")
    predict_emotion("This product is absolutely amazing, best purchase I've ever made!", distilbert_model, db_tokenizer)
    predict_emotion("Terrible quality, broke after one use. Complete waste of money.",   distilbert_model, db_tokenizer)
    predict_emotion("It works fine, does exactly what it says. Nothing special.",        distilbert_model, db_tokenizer)
    predict_emotion("I am so frustrated, this never works and support is useless.",      distilbert_model, db_tokenizer)


    # ─────────────────────────────────────────────
    # MODEL B: Bi-LSTM
    # ─────────────────────────────────────────────

    def _tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())

    def build_vocab(texts, max_vocab=30000):
        counter = Counter()
        for t in texts:
            counter.update(_tokenize(t))
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in counter.most_common(max_vocab - 2):
            vocab[word] = len(vocab)
        return vocab

    def load_glove(path, vocab, embed_dim):
        # Auto-detect actual dim from file — prevents EMBED_DIM mismatch crashes
        with open(path, 'r', encoding='utf-8') as f:
            actual_dim = len(f.readline().split()) - 1
        use_dim = actual_dim  # always use what the file actually contains
        if actual_dim != embed_dim:
            print(f"  GloVe file is {actual_dim}-d but EMBED_DIM={embed_dim} — using {actual_dim}-d.")
        embeddings = np.random.uniform(-0.1, 0.1, (len(vocab), use_dim))
        embeddings[0] = 0
        found = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                if parts[0] in vocab:
                    embeddings[vocab[parts[0]]] = np.array(parts[1:], dtype=np.float32)
                    found += 1
        print(f"  GloVe ({actual_dim}-d): {found}/{len(vocab)} vocab words matched.")
        return torch.tensor(embeddings, dtype=torch.float), use_dim


    class BiLSTMClassifier(nn.Module):
        """
          - Additive attention over all LSTM time steps
          - LayerNorm before classifier for stable training
          - Deeper projection head: 2H -> 256 -> 128 -> num_labels
            (single linear layer was compressing 768 dims in one jump)
          - GloVe pretrained embeddings (falls back to random if file absent)
        """
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
            # v7: deeper head — 2H -> 256 -> 128 -> labels
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
            x, _    = self.bilstm(self.embedding(input_ids))   # (B, T, 2H)
            mask    = (input_ids != 0)
            scores  = self.attn_fc(x).squeeze(-1)
            scores  = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            context = (weights * x).sum(dim=1)                  # (B, 2H)
            context = self.layer_norm(context)
            proj    = self.projection(self.dropout(context))
            return self.classifier(proj)


    print("\n" + "="*55)
    print("  MODEL B: Bi-LSTM  ")
    print("="*55)

    print("  Building vocabulary...")
    vocab = build_vocab(train_df['text'].tolist(), max_vocab=20000)  # 20K gives 99.3% coverage vs 30K marginal gain
    print(f"  Vocabulary size: {len(vocab):,}")

    print('  Loading embeddings...')
    if os.path.exists(GLOVE_PATH):
        glove_matrix, EMBED_DIM = load_glove(GLOVE_PATH, vocab, EMBED_DIM)
    else:
        print(f"  ⚠  GloVe file not found at '{GLOVE_PATH}' — using random init.")
        glove_matrix = None

    train_loader_lstm = DataLoader(
        LSTMDataset(train_df['text'], train_df['label'], vocab, LSTM_MAX_LEN),
        batch_size=LSTM_BATCH, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader_lstm = DataLoader(
        LSTMDataset(val_df['text'],   val_df['label'],   vocab, LSTM_MAX_LEN),
        batch_size=LSTM_BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader_lstm = DataLoader(
        LSTMDataset(test_df['text'],  test_df['label'],  vocab, LSTM_MAX_LEN),
        batch_size=LSTM_BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    bilstm_model = BiLSTMClassifier(
        vocab_size=len(vocab), embed_dim=EMBED_DIM,
        hidden_dim=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
        num_labels=NUM_LABELS, dropout=LSTM_DROPOUT,
        pretrained_embeddings=glove_matrix
    ).to(device)

    lstm_criterion = SmoothCrossEntropyLoss(smoothing=LABEL_SMOOTH, weight=class_weights)
    lstm_optimizer = torch.optim.Adam(bilstm_model.parameters(), lr=LSTM_LR)
    # linear warmup for 2 epochs then cosine decay
    # warmup prevents large random-embed gradients from destabilising early training
    def lstm_lr_lambda(epoch):
        warmup_epochs = 2
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, LSTM_EPOCHS - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress)) * (1 - 1e-5/LSTM_LR) + 1e-5/LSTM_LR
    lstm_scheduler = torch.optim.lr_scheduler.LambdaLR(lstm_optimizer, lstm_lr_lambda)

    lstm_train_losses, lstm_val_losses   = [], []
    lstm_train_metrics, lstm_val_metrics = [], []
    best_lstm_f1 = 0.0
    lstm_no_improve = 0

    print(f"\n{'─'*70}")
    print(f"  Bi-LSTM | Epochs: {LSTM_EPOCHS} | Hidden: {LSTM_HIDDEN} | "
          f"Batches/epoch: {len(train_loader_lstm)} | Patience: {LSTM_PATIENCE}")
    print(f"{'─'*70}")

    for epoch in range(LSTM_EPOCHS):
        bilstm_model.train()
        epoch_loss, ep, el = 0.0, [], []

        for batch in tqdm(train_loader_lstm,
                          desc=f"[Bi-LSTM] E{epoch+1}/{LSTM_EPOCHS} TRAIN",
                          unit='batch', leave=False):
            ids = batch['input_ids'].to(device)
            lbl = batch['labels'].to(device)
            lstm_optimizer.zero_grad()
            logits = bilstm_model(ids)
            loss   = lstm_criterion(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bilstm_model.parameters(), 1.0)
            lstm_optimizer.step()
            epoch_loss += loss.item()
            ep.extend(torch.softmax(logits, dim=1).detach().cpu().numpy())
            el.extend(lbl.cpu().numpy())

        avg_train = epoch_loss / len(train_loader_lstm)
        train_m   = compute_epoch_metrics(el, ep)

        bilstm_model.eval()
        val_loss_total, vp, vl = 0.0, [], []

        with torch.no_grad():
            for batch in tqdm(val_loader_lstm,
                              desc=f"[Bi-LSTM] E{epoch+1}/{LSTM_EPOCHS} VAL  ",
                              unit='batch', leave=False):
                ids = batch['input_ids'].to(device)
                lbl = batch['labels'].to(device)
                logits = bilstm_model(ids)
                val_loss_total += lstm_criterion(logits, lbl).item()
                vp.extend(torch.softmax(logits, dim=1).cpu().numpy())
                vl.extend(lbl.cpu().numpy())

        avg_val = val_loss_total / len(val_loader_lstm)
        val_m   = compute_epoch_metrics(vl, vp)

        lstm_train_losses.append(avg_train)
        lstm_val_losses.append(avg_val)
        lstm_train_metrics.append(train_m)
        lstm_val_metrics.append(val_m)
        print_epoch_row(epoch, LSTM_EPOCHS, avg_train, avg_val, train_m, val_m, "Bi-LSTM")

        lstm_scheduler.step()

        if val_m['macro_f1'] > best_lstm_f1:
            best_lstm_f1    = val_m['macro_f1']
            lstm_no_improve = 0
            torch.save(bilstm_model.state_dict(),
                       os.path.join(OUTPUT_DIR, 'Bi-LSTM_best.pt'))
            print(f"  ✓ Best checkpoint saved (val macro-F1: {best_lstm_f1:.4f})")
        else:
            lstm_no_improve += 1
            if lstm_no_improve >= LSTM_PATIENCE:
                print(f"  ⚑ Early stopping at epoch {epoch+1}")
                break

    print(f"{'─'*70}")
    bilstm_model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'Bi-LSTM_best.pt')))
    print(f"  Reloaded best checkpoint (val macro-F1: {best_lstm_f1:.4f})")

    plot_curves(lstm_train_losses, lstm_val_losses,
                lstm_train_metrics, lstm_val_metrics, "Bi-LSTM")

    metrics_bilstm = evaluate_model(bilstm_model, test_loader_lstm, "Bi-LSTM", is_lstm=True)
    torch.save(bilstm_model.state_dict(), os.path.join(OUTPUT_DIR, 'bilstm_textsense.pt'))
    print(f"  Saved: {OUTPUT_DIR}/bilstm_textsense.pt")

    print("\n--- Bi-LSTM Inference Test ---")
    predict_emotion_lstm("This product is absolutely amazing, best purchase I've ever made!", bilstm_model, vocab)
    predict_emotion_lstm("Terrible quality, broke after one use. Complete waste of money.",   bilstm_model, vocab)
    predict_emotion_lstm("It works fine, does exactly what it says. Nothing special.",        bilstm_model, vocab)
    predict_emotion_lstm("I am so frustrated, this never works and support is useless.",      bilstm_model, vocab)


    # ─────────────────────────────────────────────
    # MODEL C: RoBERTa
    # ─────────────────────────────────────────────

    class RoBERTaClassifier(nn.Module):
        """
        v3: projection head 768 → 256 → 4 (now actually wired in forward()).
        LR reduced to 1e-5 backbone / 5e-5 head via layer-wise decay.
        """
        def __init__(self, num_labels, dropout=DROPOUT):
            super().__init__()
            self.roberta    = RobertaModel.from_pretrained('roberta-base')
            hidden          = self.roberta.config.hidden_size   # 768
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


    print("\n" + "="*55)
    print("  MODEL C: RoBERTa  ")
    print("="*55)

    rb_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    train_loader_rb = DataLoader(
        EmotionDataset(train_df['text'], train_df['label'], rb_tokenizer, MAX_LEN),
        batch_size=RB_BATCH, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader_rb = DataLoader(
        EmotionDataset(val_df['text'],   val_df['label'],   rb_tokenizer, MAX_LEN),
        batch_size=RB_BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader_rb = DataLoader(
        EmotionDataset(test_df['text'],  test_df['label'],  rb_tokenizer, MAX_LEN),
        batch_size=RB_BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    roberta_model   = RoBERTaClassifier(NUM_LABELS).to(device)
    roberta_model   = train_transformer(
        roberta_model, train_loader_rb, val_loader_rb,
        "RoBERTa",
        epochs=RB_EPOCHS, lr_backbone=RB_LR_BACKBONE, lr_head=RB_LR_HEAD,
        warmup=RB_WARMUP, patience=RB_PATIENCE
    )
    metrics_roberta = evaluate_model(roberta_model, test_loader_rb, "RoBERTa")

    torch.save(roberta_model.state_dict(),
               os.path.join(OUTPUT_DIR, 'roberta_textsense.pt'))
    print(f"  Saved: {OUTPUT_DIR}/roberta_textsense.pt")

    print("\n--- RoBERTa Inference Test ---")
    predict_emotion("This product is absolutely amazing, best purchase I've ever made!", roberta_model, rb_tokenizer)
    predict_emotion("Terrible quality, broke after one use. Complete waste of money.",   roberta_model, rb_tokenizer)
    predict_emotion("It works fine, does exactly what it says. Nothing special.",        roberta_model, rb_tokenizer)
    predict_emotion("I am so frustrated, this never works and support is useless.",      roberta_model, rb_tokenizer)


    # ─────────────────────────────────────────────
    # FINAL COMPARISON SUMMARY
    # ─────────────────────────────────────────────

    print("\n" + "="*70)
    print("  TEXTSENSE — FINAL COMPARATIVE RESULTS SUMMARY")
    print("="*70)

    summary = pd.DataFrame({
        'Model'         : ['DistilBERT', 'Bi-LSTM', 'RoBERTa'],
        'Accuracy'      : [f"{metrics_distilbert['subset_accuracy']:.4f}",
                           f"{metrics_bilstm['subset_accuracy']:.4f}",
                           f"{metrics_roberta['subset_accuracy']:.4f}"],
        'Precision (M)' : [f"{metrics_distilbert['precision']:.4f}",
                           f"{metrics_bilstm['precision']:.4f}",
                           f"{metrics_roberta['precision']:.4f}"],
        'Recall (M)'    : [f"{metrics_distilbert['recall']:.4f}",
                           f"{metrics_bilstm['recall']:.4f}",
                           f"{metrics_roberta['recall']:.4f}"],
        'Macro F1'      : [f"{metrics_distilbert['macro_f1']:.4f}",
                           f"{metrics_bilstm['macro_f1']:.4f}",
                           f"{metrics_roberta['macro_f1']:.4f}"],
        'Weighted F1'   : [f"{metrics_distilbert['weighted_f1']:.4f}",
                           f"{metrics_bilstm['weighted_f1']:.4f}",
                           f"{metrics_roberta['weighted_f1']:.4f}"],
        '≥80%?'         : [
            '✓' if metrics_distilbert['subset_accuracy'] >= 0.80 else '✗',
            '✓' if metrics_bilstm['subset_accuracy']     >= 0.80 else '✗',
            '✓' if metrics_roberta['subset_accuracy']    >= 0.80 else '✗',
        ],
    })
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 5, figsize=(22, 6))
    fig.suptitle('TextSense v8 — Model Comparison (Test Set, 4-Class)',
                 fontsize=15, fontweight='bold')

    models_names  = ['DistilBERT', 'Bi-LSTM', 'RoBERTa']
    bar_colors    = ['#4C72B0', '#55A868', '#C44E52']
    metrics_to_plot = [
        ('subset_accuracy', 'Accuracy',       'Accuracy'),
        ('precision',       'Macro Precision','Precision'),
        ('recall',          'Macro Recall',   'Recall'),
        ('macro_f1',        'Macro F1',       'F1'),
        ('weighted_f1',     'Weighted F1',    'Weighted F1'),
    ]
    all_metrics = [metrics_distilbert, metrics_bilstm, metrics_roberta]

    for ax, (key, title, ylabel) in zip(axes, metrics_to_plot):
        vals = [m[key] for m in all_metrics]
        bars = ax.bar(models_names, vals, color=bar_colors)
        ax.axhline(0.80, color='green', linestyle='--', alpha=0.7, label='80% target')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, min(max(vals) * 1.18, 1.0))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150)
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR}/model_comparison.png")
    print("\n  All training complete. Models and plots saved to:", OUTPUT_DIR)


if __name__ == '__main__':
    main()