# Accuracy Tables - TESS (test, n=560)

| Model | Accuracy | Macro-F1 (approx) |
|---|---|---|
| Speech-only (MFCC + BiLSTM + Attention) | 0.8768 | 0.88 |
| Text-only (Tokens + BiLSTM + Attention) | 0.8321 | 0.83 |
| Multimodal Fusion (Late + Cross-Attention) | 0.8500 | 0.85 |

Fusion helps most for fear vs surprise and happy vs neutral where one modality alone is ambiguous. See Report for per-emotion analysis.
