# BCI Preprocessing with RPCA

Robust preprocessing pipeline for Brain-Computer Interface (BCI) EEG signals using **Robust Principal Component Analysis (RPCA)**. The project applies RPCA to decompose EEG data into low-rank (signal structure) and sparse (noise/artifacts) components, then evaluates classification accuracy across multiple preprocessing strategies.

## Techniques

- **RPCA filtering** — separates EEG signals into low-rank (L) and sparse (S) components using an augmented Lagrangian method
- **Common Average Reference (CAR)** — baseline spatial filter for comparison
- **Spectrogram features** — time-frequency decomposition via STFT for frequency-domain classification
- **SVM / LDA classification** — trained on preprocessed features with stratified train/test splits

## Project Structure

```
code/
├── rpca.py                  # Core RPCA implementation
├── spectrogram_gpu.py       # GPU-accelerated spectrogram (CuPy)
├── spectrogram_pytorch.py   # PyTorch spectrogram with batch support
├── sandbox/                 # Experimental notebooks
│   ├── rpca-bci.ipynb
│   ├── rpca-bci-patients.ipynb
│   ├── car-filter.ipynb
│   └── ...
├── results/                 # CSV accuracy results
└── data/                    # EEG datasets (S1.mat–S32.mat)
images/                      # Generated plots and figures
notes/                       # Text notes and intermediate results
test/                        # RPCA image tutorials and demos
rpca_parameter_selection.md  # Parameter optimization methodology
requirements.txt             # Python dependencies
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preprocessing Methods Compared

| Method | Description |
|---|---|
| **RPCA_L** | Low-rank component from RPCA |
| **RPCA_S** | Sparse component from RPCA |
| **CAR_RPCA_L** | CAR filtering followed by RPCA low-rank extraction |
| **CAR_RPCA_S** | CAR filtering followed by RPCA sparse extraction |
