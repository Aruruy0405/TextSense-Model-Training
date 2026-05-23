# TextSense — Emotion Recognition from User Reviews

**_LOCALLY TRAINED_** Models: DistilBERT | Bi-LSTM | RoBERTa

Task: 4-class emotion classification (anger | sadness | joy | neutral)  

Dataset: GoEmotions → 4-class aggregated

 Results

| Model      | Accuracy | Macro F1 | ≥80%? |
|------------|----------|----------|-------|
| DistilBERT | 0.8343   | 0.8315   | ✓     |
| Bi-LSTM    | 0.8017   | 0.8028   | ✓     |
| RoBERTa    | 0.8087   | 0.8047   | ✓     |

 Files
- `TextSense_Training_Local.py` — full training script
- `TextSense_Inference.py` — load saved models and run evaluation
- `textsense_outputs/` — training curves, confusion matrices, comparison plots


 Large Files (not in repo — available on request)
- Model weights (`.pt`) — stored on Google Drive: https://drive.google.com/drive/folders/1rrD-kJKJEk0C2ZT6w53YHELxIishvgua?usp=sharing
- `goemotions(FINAL).csv` — you can download it here: https://huggingface.co/datasets/mrm8488/goemotions/blob/bd3ed9a7817b7a9f0742593ac893ab7b2dc2b996/goemotions.csv or stored in Google Drive: https://drive.google.com/drive/folders/1rrD-kJKJEk0C2ZT6w53YHELxIishvgua?usp=sharing
- `glove.6B.200d.txt` — GloVe embeddings

Setup:
Step 1: Check your Python version
Open Command Prompt and run:
bashpython --version
You need Python 3.9, 3.10, or 3.11. If you don't have it, download from python.org. Avoid Python 3.12+ as PyTorch has limited support for it.

Step 2: Check your CUDA version
Open Command Prompt and run:
bashnvidia-smi
Look for CUDA Version in the top right corner. It should say 12.x or 11.x. This determines which PyTorch build to install.

Step 3: Install PyTorch with CUDA
bash For CUDA 12.1 (most common on modern drivers)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

If your CUDA is 11.8 instead, use this:
Open Command Prompt and run:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Step 4: Install everything else
Open Command Prompt and run:
bashpip install transformers scikit-learn pandas matplotlib seaborn tqdm

Step 5: Download GloVe manually (needed for Bi-LSTM)
Download from: https://nlp.stanford.edu/data/glove.6B.zip
Extract it and put glove.6B.200d.txt in the same folder as your script.

------------------------------------------------------------------------
Single Folder Structure:

TextSense_Training_Local.py 

TextSense_Inference.py

goemotions(FINAL).csv

glove.6B.200d.txt ← extract from the zip you download

 How to Run Inference
1. Place `.pt` files in `textsense_outputs/`
2. Place `goemotions(FINAL).csv` in root folder
3. Run: `python TextSense_Inference.py`
