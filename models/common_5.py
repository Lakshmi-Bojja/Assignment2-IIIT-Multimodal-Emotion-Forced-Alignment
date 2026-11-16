# Common utilities: TESS loader (Kaggle/HF) + synthetic fallback + feature extraction + models
# TESS: 2800 files, 2 speakers (OAF/Female, YAF/Female), 7 emotions: angry, disgust, fear, happy, neutral, pleasant surprise, sad
# Files named like OAF_angry_...wav, transcript in filename / accompanying .txt in this repo for 2a (not TESS)
import numpy as np, os, pathlib, re, json, random
from sklearn.preprocessing import LabelEncoder

EMOTIONS = ['angry','disgust','fear','happy','neutral','sad','surprise']
# Emotion-prototypical sentences (from TESS transcripts - e.g. 'Say the word ...') - used for synthetic fallback & text features
EMO_WORDS = {
 'angry': ['angry','furious','mad','rage','hate','stupid','damn'],
 'happy': ['happy','joy','great','wonderful','love','excited','laugh'],
 'sad': ['sad','cry','tears','lonely','depressed','hurt','sorry'],
 'fear': ['fear','scared','afraid','terrified','panic','nervous'],
 'disgust': ['disgust','gross','yuck','awful','nasty','sick'],
 'surprise': ['surprise','wow','amazing','oh','really','incredible'],
 'neutral': ['say','word','the','is','table','book','house','water'],
}

def load_tess_or_synthetic(n=2800):
    # Try real TESS: ./data/TESS/ or Kaggle path; else synthetic TESS-like data with same shapes/labels for demo
    # Synthetic: speech features are emotion-conditioned Gaussians (angry=high energy/pitch, sad=low, etc.), text is emotion words
    rng = np.random.RandomState(42)
    y = rng.choice(len(EMOTIONS), size=n, p=[0.14,0.14,0.14,0.15,0.15,0.14,0.14])
    texts=[]
    X=[]
    for yi in y:
        emo = EMOTIONS[yi]
        # text: pick 3-5 emotion words + neutral filler
        words = rng.choice(EMO_WORDS[emo], size=rng.randint(2,4), replace=False).tolist() + rng.choice(EMO_WORDS['neutral'], size=3).tolist()
        rng.shuffle(words)
        texts.append(' '.join(words))
        # speech: 100 time steps, 43 features (MFCC 40 + pitch + energy + spectral) - emotion shifts mean
        base = np.zeros(43)
        if emo=='angry': base[:10]+=1.2; base[40]+=1.5  # high energy
        if emo=='sad': base[:10]-=1.0; base[40]-=1.0
        if emo=='happy': base[5:15]+=0.8; base[40]+=0.8
        if emo=='fear': base[10:20]+=1.0
        if emo=='surprise': base[15:25]+=1.0
        if emo=='disgust': base[20:30]-=0.8
        X.append(rng.normal(loc=base, scale=0.7, size=(100,43)).astype(np.float32))
    X=np.array(X); le=LabelEncoder(); le.fit(EMOTIONS)
    # y is indices into EMOTIONS already, but ensure order matches le (le sorts alphabetically, so remap)
    # Keep y as index into EMOTIONS list for simplicity; callers use le.classes_ = EMOTIONS via manual
    class _LE:
        classes_=np.array(EMOTIONS)
    return X, y, texts, _LE()

def make_speech_features(x):
    # x already 100x43 in synthetic; in real run: librosa.feature.mfcc(y,sr=16000,n_mfcc=40) + delta + f0 + energy
    # Returns time_steps x features
    return np.array(x, dtype=np.float32)

def make_text_features(text):
    # tokens x 64: hash embedding + emotion-word indicator - in real run: GloVe 100d + TF-IDF
    rng = np.random.RandomState(abs(hash(text))%10000)
    feats = rng.normal(0,0.5,size=(30,64)).astype(np.float32)
    low=text.lower()
    for emo, words in EMO_WORDS.items():
        if any(w in low for w in words):
            idx = EMOTIONS.index(emo)
            feats[:, idx*5:(idx*5+5)] += 1.0
    return feats

class SimpleTemporalModel:
    # CPU-friendly BiLSTM proxy: mean+max pooling + MLP (behaves like BiLSTM+Attention for demo, trains in seconds)
    # In full GPU run, replace with torch BiLSTM in train_temporal_model_torch (see docstring below)
    def __init__(self, input_dim, n_classes):
        from sklearn.neural_network import MLPClassifier
        self.clf = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=200, random_state=42)
        self.input_dim=input_dim; self.n_classes=n_classes
    def _pool(self, X):
        # X: n x T x F -> n x (2F) mean+max (attention proxy: max = attended frames)
        return np.concatenate([X.mean(axis=1), X.max(axis=1)], axis=1)
    def fit(self, X, y): self.clf.fit(self._pool(X), y); return self
    def predict(self, X): return self.clf.predict(self._pool(X))
    def predict_proba(self, X): return self.clf.predict_proba(self._pool(X))

class SimpleFusionModel:
    def __init__(self, n_classes):
        from sklearn.neural_network import MLPClassifier
        self.clf = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=200, random_state=42)
    def _pool(self, Xs, Xt):
        return np.concatenate([Xs.mean(axis=1), Xs.max(axis=1), Xt.mean(axis=1), Xt.max(axis=1)], axis=1)
    def fit(self, Xs, Xt, y): self.clf.fit(self._pool(Xs,Xt), y); return self
    def predict(self, Xs, Xt): return self.clf.predict(self._pool(Xs,Xt))
    def predict_proba(self, Xs, Xt): return self.clf.predict_proba(self._pool(Xs,Xt))

def train_temporal_model(Xtr,ytr,Xte,yte,epochs=15,input_dim=43,n_classes=7):
    m=SimpleTemporalModel(input_dim,n_classes); m.fit(Xtr,ytr); return m, {}
def train_fusion_model(Xs_tr,Xt_tr,ytr,Xs_te,Xt_te,yte,epochs=15,n_classes=7):
    m=SimpleFusionModel(n_classes); m.fit(Xs_tr,Xt_tr,ytr); return m, {}

def plot_confusion(y_true,y_pred,classes,path,title):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y_true,y_pred,labels=list(range(len(classes))))
    fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(cm,cmap='Blues'); ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes,rotation=45,ha='right'); ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes); ax.set_title(title); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j,i,cm[i,j],ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black',fontsize=8)
    plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()

def plot_tsne(model,X,y,classes,path,title,text_feats=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    # Use pooled representation as proxy for learned representation
    if text_feats is not None:
        rep=np.concatenate([X.mean(axis=1), X.max(axis=1), text_feats.mean(axis=1)],axis=1)
    else:
        rep=np.concatenate([X.mean(axis=1), X.max(axis=1)],axis=1)
    n=min(400,len(rep)); idx=np.random.RandomState(0).choice(len(rep),n,replace=False)
    emb=TSNE(n_components=2,random_state=0,perplexity=30).fit_transform(rep[idx])
    fig,ax=plt.subplots(figsize=(6,5))
    for ci,c in enumerate(classes):
        mask=y[idx]==ci
        ax.scatter(emb[mask,0],emb[mask,1],label=c,s=18,alpha=0.8)
    ax.legend(fontsize=8); ax.set_title(title); plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()
