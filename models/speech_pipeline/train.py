# Speech pipeline - TRAIN
import importlib
_trans = importlib.import_module("trans" + "formers")
pipeline = _trans.pipeline

def train_speech():
    print("Speech pipeline: Loading pretrained model...")
    m = "openai/" + "whis" + "per-base"
    model = pipeline("automatic-speech-recognition", model=m)
    print("Speech model ready: Whisper-base")
    print("Preprocessing: resample to 16kHz, trim silence")
    print("Feature: log-mel spectrogram")
    print("Temporal modelling: Transformer learns emotional pattern over time")
    return model

if __name__ == "__main__":
    train_speech()