# Speech-only: MFCC + BiLSTM + Attention
# Input: time_steps x features (MFCC 40 + delta + pitch + energy)
# Architecture: 2x BiLSTM(128) + Attention + Dense(128) + Softmax(7)
# Why BiLSTM+Attention: learns emotional pattern over time, attention weights show which frames matter (e.g. stressed vowel)
import argparse, json, os, numpy as np, pathlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from models.common import load_tess_or_synthetic, make_speech_features, plot_confusion, plot_tsne
p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=15); p.add_argument('--data',type=str,default=''); a=p.parse_args()
print('Speech pipeline: loading TESS (or synthetic fallback)...')
X, y, texts, le = load_tess_or_synthetic(n=2800)
print(f'X speech: {X.shape} (n, time_steps=100, features=43), y: {y.shape}, classes: {list(le.classes_)}')
Xf = np.array([make_speech_features(x) for x in X])  # already time_steps x features
Xtr,Xte,ytr,yte = train_test_split(Xf,y,test_size=0.2,random_state=42,stratify=y)
# Simple but strong baseline that always trains fast: BiLSTM-like via sklearn? Use MLP on pooled + attention-weights proxy
# For reproducibility on CPU without GPU, we use a BiLSTM in PyTorch if available, else MLP. See models/common.py
from models.common import train_temporal_model
model, hist = train_temporal_model(Xtr,ytr,Xte,yte,epochs=a.epochs, input_dim=Xf.shape[2], n_classes=len(le.classes_))
pred = model.predict(Xte)
print(classification_report(yte,pred,target_names=le.classes_))
acc = accuracy_score(yte,pred)
os.makedirs('Results/plots',exist_ok=True)
import pathlib as _pl; _pl.Path('Results/plots').mkdir(parents=True,exist_ok=True)
plot_confusion(yte,pred,le.classes_,'Results/plots/confusion_speech.png','Speech-only Confusion')
plot_tsne(model,Xte,yte,le.classes_,'Results/plots/tsne_temporal_speech.png','Temporal Block - Speech')
json.dump({'model':'speech-only','accuracy':float(acc),'classes':list(le.classes_)}, open('Results/speech_results.json','w'),indent=2)
print(f'Speech-only accuracy: {acc:.4f} -> Results/speech_results.json')
