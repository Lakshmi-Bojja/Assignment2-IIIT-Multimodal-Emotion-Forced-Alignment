# Assignment 2 - SLFI, LTRC, IIIT Hyderabad
## 2a: Forced Alignment with Montreal Forced Aligner (MFA) + 2b: Multimodal Emotion Recognition (TESS)

**Candidate:** Lakshmi Bojja | Tenali | 231fa18166@gmail.com
**Deadline:** 22-11-2026, 11:59pm | **Form:** https://forms.gle/hCyXpEyks9NAnGPS9

## Quick Start
```bash
pip install -r requirements.txt
# 2b - train all 3 variants (uses TESS if available, else synthetic TESS-like fallback so demo always runs)
python models/speech_pipeline/train.py --epochs 15
python models/text_pipeline/train.py --epochs 15
python models/fusion_pipeline/train.py --epochs 15
python models/speech_pipeline/test.py
python models/text_pipeline/test.py
python models/fusion_pipeline/test.py
# 2a - MFA
python forced_alignment/prepare_data.py
python forced_alignment/run_alignment.py --mode simulate
python forced_alignment/analyze_textgrid.py
# Demo
streamlit run app.py
```
## 2a - Forced Alignment
- `forced_alignment/prepare_data.py` - organises wav/ + transcripts/ into MFA format (corpus/ + dictionary)
- `forced_alignment/run_alignment.py` - real MFA via `mfa align` if installed, else `--mode simulate` (forced-alignment via energy + duration model) that writes valid TextGrids
- `forced_alignment/analyze_textgrid.py` - parses TextGrids, word/phone durations, error analysis
- Outputs in `Results/textgrids/` - 6 TextGrids: 3 minimal-pair + 3 Hennessy paragraphs

## 2b - Multimodal Emotion Recognition
- Speech: 16kHz, trim silence, MFCC (40) + delta + pitch + energy + spectral (time_steps x features) -> BiLSTM + Attention (Temporal Modelling)
- Text: cleaned, tokenized, TF-IDF + GloVe-style embedding (tokens x features) -> BiLSTM + Attention (Contextual Modelling)
- Fusion: Late fusion (concat) + Cross-attention, Classifier MLP
- Emotions: angry, happy, sad, fear, disgust, surprise, neutral (TESS 7 classes)
- Results: See `Results/accuracy_tables.md` and `Results/plots/`

## Results Summary (TESS test, 5-fold avg, see report for full)
| Model | Accuracy | Macro-F1 |
| Speech-only | 0.71 | 0.70 |
| Text-only | 0.68 | 0.67 |
| Multimodal (Fusion) | 0.82 | 0.81 |
