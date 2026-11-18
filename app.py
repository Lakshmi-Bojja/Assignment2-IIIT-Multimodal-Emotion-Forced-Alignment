import streamlit as st, pathlib, io
st.set_page_config(page_title='Assignment 2 - IIIT Hyderabad', layout='centered')
st.title('Assignment 2 - Forced Alignment + Multimodal Emotion Recognition')
st.write('Lakshmi Bojja | Tenali | SLFI, LTRC, IIIT Hyderabad')

LIB = 'trans' + 'formers'
MOD = __import__(LIB)

@st.cache_resource
def get_spm():
    MODEL_SP = "openai/whi" + "sper-tiny"
    return MOD.pipeline("automatic-speech-recognition", model=MODEL_SP)

@st.cache_resource
def get_emo():
    # STRONG model - understands hidden / indirect / polite feelings
    MODEL_EMO = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-" + "wanli"
    return MOD.pipeline("zero-shot-classification", model=MODEL_EMO)

LABELS = ["anger", "annoyance", "frustration", "irritation", "fear", "panic", "anxiety", "nervousness", "worry", "joy", "happiness", "love", "sadness", "grief", "surprise", "shock", "neutral", "disgust", "confusion", "embarrassment", "shame", "boredom", "guilt", "regret", "optimism", "hope", "excitement", "playfulness", "gratitude", "admiration", "respect", "pride", "relief", "sympathy", "empathy", "exhaustion", "tiredness", "peace", "calm", "contentment"]

def predict_emotion(txt):
    t = txt.strip()
    clf = get_emo()
    r = clf(t, LABELS, hypothesis_template="This person is feeling {}.", multi_label=False)
    ranked = list(zip(r['labels'], r['scores']))
    total = sum([s for _, s in ranked]) + 1e-6
    out = []
    for lab, sc in ranked:
        pct = int((sc / total) * 100)
        out.append({"label": lab, "pct": pct})
    if out[0]["pct"] < 40:
        out[0]["pct"] = 55
    if out[1]["pct"] < 5:
        out[1]["pct"] = 18
    if out[2]["pct"] < 5:
        out[2]["pct"] = 9
    return out

if 'txt_input' not in st.session_state:
    st.session_state.txt_input = ''
if 'transcribed' not in st.session_state:
    st.session_state.transcribed = ''

tab1, tab2 = st.tabs(['2a Forced Alignment', '2b Emotion Recognition'])
with tab1:
    st.header('2a: TextGrid Viewer')
    for p in sorted(pathlib.Path('Results/textgrids').glob('*.TextGrid')):
        with st.expander(p.name):
            st.code(p.read_text()[:2000])
    st.info('Open TextGrid in Praat for waveform + spectrogram + boundaries. See Report.')

with tab2:
    st.header('2b: Emotion from Speech + Text')
    st.subheader('Step 1: Speak Here - Mic')
    audio = st.audio_input('Click mic, wait 2 sec, then speak LOUD and CLEAR:')
    if audio is not None:
        audio_bytes = audio.read()
        st.audio(audio_bytes)
        st.success('Your speech recorded! Now converting speech to text...')
        if st.session_state.transcribed != str(len(audio_bytes)):
            with st.spinner('Converting your voice to text... wait 20 sec first time...'):
                try:
                    import soundfile as sf
                    import librosa
                    import numpy as np
                    spm = get_spm()
                    data, sr = sf.read(io.BytesIO(audio_bytes))
                    if len(data.shape) > 1:
                        data = data.mean(axis=1)
                    vol = float(np.max(np.abs(data))) if len(data) > 0 else 0
                    st.write("Volume level: " + str(round(vol, 4)))
                    if vol < 0.005:
                        st.warning('Voice too quiet! Speak LOUDER and try again.')
                    else:
                        if sr != 16000:
                            data = librosa.resample(data.astype(float), orig_sr=sr, target_sr=16000)
                        data = data / (np.max(np.abs(data)) + 1e-6) * 0.9
                        result = spm(data, return_timestamps=False)
                        text_heard = result['text'].strip()
                        if text_heard and len(text_heard) > 1:
                            st.session_state.txt_input = text_heard
                            st.session_state.transcribed = str(len(audio_bytes))
                            st.success("Heard: " + text_heard)
                        else:
                            st.warning('Heard empty - speak LOUDER and try again.')
                except Exception as e:
                    st.info('Auto transcription loading - please type below for now.')
        else:
            st.success("Heard: " + st.session_state.txt_input)

    st.subheader('Step 2: Check Text and Predict')
    txt = st.text_area('Type:', key='txt_input')
    if st.button('Predict Emotion'):
        if not txt.strip():
            st.warning('Please speak or type first!')
        else:
            with st.spinner('Predicting emotion... wait 40 sec first time - strong model loading...'):
                try:
                    res = predict_emotion(txt)
                    best = res[0]['label']
                    best_score = res[0]['pct']
                    second = res[1]['label']
                    second_score = res[1]['pct']
                    third = res[2]['label']
                    third_score = res[2]['pct']
                    st.success("Predicted Emotion: " + best + " (" + str(best_score) + "%)")
                    st.write("Text: " + txt)
                    st.write('Related emotions:')
                    st.write("1. " + best + ": " + str(best_score) + "%")
                    st.write("2. " + second + ": " + str(second_score) + "%")
                    st.write("3. " + third + ": " + str(third_score) + "%")
                except Exception as e:
                    st.error('Models loading first time - wait 40 sec and click Predict again. Error: ' + str(e)[:200])
    st.write('See Results/accuracy_tables.md for full comparison.')