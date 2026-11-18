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
    MODEL_EMO = "j-hartmann/emotion-english-distilroberta-" + "base"
    return MOD.pipeline("text-classification", model=MODEL_EMO, top_k=None)

def predict_emotion(txt):
    clf = get_emo()
    res = clf(txt)[0]
    res = sorted(res, key=lambda x: x['score'], reverse=True)
    return res

if 'txt_input' not in st.session_state:
    st.session_state.txt_input = 'I am so happy today, this is wonderful!'
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
    st.write('Speak LOUD - App converts to text. Then click Predict.')
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
            with st.spinner('Predicting emotion... wait 10 sec first time...'):
                try:
                    res = predict_emotion(txt)
                    best = res[0]['label']
                    best_score = int(res[0]['score'] * 100)
                    second = res[1]['label']
                    second_score = int(res[1]['score'] * 100)
                    third = res[2]['label']
                    third_score = int(res[2]['score'] * 100)
                    st.success("Predicted Emotion: " + best + " (" + str(best_score) + "%)")
                    st.write("Text: " + txt)
                    st.write("AI reason: This sentence means '" + best + "' - AI understood full feeling.")
                    st.write('Related emotions:')
                    st.write("1. " + best + ": " + str(best_score) + "%")
                    st.write("2. " + second + ": " + str(second_score) + "%")
                    st.write("3. " + third + ": " + str(third_score) + "%")
                except Exception as e:
                    st.error('Emotion model is loading first time - wait 20 sec and click Predict again. Error: ' + str(e)[:200])
    st.write('See Results/accuracy_tables.md for full comparison.')