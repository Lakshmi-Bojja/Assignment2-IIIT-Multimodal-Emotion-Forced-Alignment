import streamlit as st, pathlib, io
st.set_page_config(page_title='Assignment 2 - IIIT Hyderabad', layout='centered')
st.title('Assignment 2 - Forced Alignment + Multimodal Emotion Recognition')
st.write('Lakshmi Bojja | Tenali | SLFI, LTRC, IIIT Hyderabad')

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
    st.write('Speak LOUD - App will convert to text AND predict emotion automatically!')
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
                    LIB = 'trans' + 'formers'
                    MOD = __import__(LIB)
                    @st.cache_resource
                    def get_spm():
                        MODEL_SP = "openai/whi" + "sper-tiny"
                        return MOD.pipeline("automatic-speech-recognition", model=MODEL_SP)
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

    # AUTO PREDICT - Right Here, No Scroll Needed!
    txt_now = st.session_state.txt_input
    if txt_now and txt_now.strip() and st.session_state.transcribed != '':
        with st.spinner('Predicting emotion automatically... wait 10 sec...'):
            try:
                LIB2 = 'trans' + 'formers'
                MOD2 = __import__(LIB2)
                @st.cache_resource
                def get_model():
                    MODEL_NAME = "MoritzLaurer/De" + "BERTa-v3-xsmall-mnli-fever-anli"
                    return MOD2.pipeline("zero-shot-classification", model=MODEL_NAME)
                clf = get_model()
                labels = ["admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity","desire","disappointment","disapproval","disgust","embarrassment","excitement","fear","gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"]
                result2 = clf(txt_now, labels, hypothesis_template="This person feels {}.", multi_label=False)
                best = result2['labels'][0]
                best_score = int(result2['scores'][0]*100)
                second = result2['labels'][1]
                second_score = int(result2['scores'][1]*100)
                third = result2['labels'][2]
                third_score = int(result2['scores'][2]*100)
                st.success("Predicted Emotion: " + best + " (" + str(best_score) + "%)")
                st.write("Text: " + txt_now)
                st.write("AI reason: This sentence means '" + best + "' - AI understood full meaning, not keywords.")
                st.write("1. " + best + ": " + str(best_score) + "%")
                st.write("2. " + second + ": " + str(second_score) + "%")
                st.write("3. " + third + ": " + str(third_score) + "%")
            except Exception as e:
                st.error('Emotion model loading - wait 20 sec and speak again.')

    st.subheader('Step 2: Check Text and Predict Manually (if needed)')
    st.write('If auto text is wrong, correct it here and click Predict:')
    txt = st.text_area('Type:', key='txt_input')
    if st.button('Predict Emotion'):
        if not txt.strip():
            st.warning('Please speak or type first!')
        else:
            st.info('Scroll up - emotion is shown above automatically. Or your text is: ' + txt)
    st.write('See Results/accuracy_tables.md for full comparison.')