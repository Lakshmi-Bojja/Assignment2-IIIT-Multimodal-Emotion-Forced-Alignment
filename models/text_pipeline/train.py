# Text-only: tokens x features -> BiLSTM + Attention (Contextual Modelling)
# Why: learns emotional meaning across tokens in context, e.g. 'not happy' vs 'happy', negation, intensifiers
import argparse, json, os, numpy as np, pathlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from models.common import load_tess_or_synthetic, plot_confusion, plot_tsne, train_temporal_model
from models.common import make_text_features

p = argparse.ArgumentParser(); p.add_argument('--epochs', type=int, default=15); a = p.parse_args()

X, y, texts, le = load_tess_or_synthetic(n=2800)

# Text features: TF-IDF + embedding-like (tokens x 64) simulated from texts - in real run, uses GloVe via models/common
Xt = np.array([make_text_features(t) for t in texts])
print(f'X text: {Xt.shape} (n, tokens=30, features=64)')

Xtr, Xte, ytr, yte = train_test_split(Xt, y, test_size=0.2, random_state=42, stratify=y)

model, hist = train_temporal_model(Xtr, ytr, Xte, yte, epochs=a.epochs, input_dim=Xt.shape[2], n_classes=len(le.classes_))

pred = model.predict(Xte)
print(classification_report(yte, pred, target_names=le.classes_))
acc = accuracy_score(yte, pred)

os.makedirs('Results/plots', exist_ok=True)
plot_confusion(yte, pred, le.classes_, 'Results/plots/confusion_text.png', 'Text-only Confusion')
plot_tsne(model, Xte, yte, le.classes_, 'Results/plots/tsne_contextual_text.png', 'Contextual Block - Text')
json.dump({'model': 'text-only', 'accuracy': float(acc)}, open('Results/text_results.json', 'w'), indent=2)
print(f'Text-only accuracy: {acc:.4f}')