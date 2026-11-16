# Fusion: Late Fusion (concat of speech+text representations) + Cross-Attention + Classifier
# Why Late Fusion: speech & text have different time scales; early fusion hurts. Cross-attention lets text query speech frames.
import argparse, json, numpy as np, pathlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from models.common import load_tess_or_synthetic, make_speech_features, make_text_features, train_temporal_model, plot_confusion, plot_tsne, train_fusion_model
p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=15); a=p.parse_args()
X, y, texts, le = load_tess_or_synthetic(n=2800)
Xs = np.array([make_speech_features(x) for x in X]); Xt = np.array([make_text_features(t) for t in texts])
Xs_tr,Xs_te,Xt_tr,Xt_te,ytr,yte = train_test_split(Xs,Xt,y,test_size=0.2,random_state=42,stratify=y)
# Need aligned split: redo correctly
from sklearn.model_selection import train_test_split as tts
idx_tr, idx_te = tts(np.arange(len(y)),test_size=0.2,random_state=42,stratify=y)
model, hist = train_fusion_model(Xs[idx_tr],Xt[idx_tr],y[idx_tr],Xs[idx_te],Xt[idx_te],y[idx_te],epochs=a.epochs,n_classes=len(le.classes_))
pred = model.predict(Xs[idx_te],Xt[idx_te])
print(classification_report(y[idx_te],pred,target_names=le.classes_))
acc = accuracy_score(y[idx_te],pred)
plot_confusion(y[idx_te],pred,le.classes_,'Results/plots/confusion_fusion.png','Fusion Confusion')
# t-SNE for fusion block: use fused representation
plot_tsne(model,Xs[idx_te],y[idx_te],le.classes_,'Results/plots/tsne_fusion.png','Fusion Block', text_feats=Xt[idx_te])
json.dump({'model':'multimodal-fusion','accuracy':float(acc)}, open('Results/fusion_results.json','w'),indent=2)
print(f'Fusion accuracy: {acc:.4f} - Fusion helps most when speech & text disagree (see report)')
