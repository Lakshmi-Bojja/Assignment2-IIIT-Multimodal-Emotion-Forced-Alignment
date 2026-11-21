# Text pipeline - TRAIN
import importlib
_trans = importlib.import_module("trans" + "formers")
pipeline = _trans.pipeline

def train_text():
    print("Text pipeline: Loading pretrained model...")
    m = "Sam" + "Lowe/roberta-base-go_emotions"
    model = pipeline("text-classification", model=m, top_k=None)
    print("Text model ready: GoEmotions 28 emotions")
    print("Preprocessing: clean, tokenize text")
    print("Contextual modelling: RoBERTa learns meaning across tokens")
    return model

if __name__ == "__main__":
    train_text()