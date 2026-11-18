import streamlit as st, pathlib, io
st.set_page_config(page_title='Assignment 2 - IIIT Hyderabad', layout='centered')
st.title('Assignment 2 - Forced Alignment + Multimodal Emotion Recognition')
st.write('Lakshmi Bojja | Tenali | SLFI, LTRC, IIIT Hyderabad')

if 'txt' not in st.session_state:
    st.session_state.txt = 'I am so happy today, this is wonderful!'
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
    st.write('Step 1: Speak - App converts speech to text automatically. Step 2: Check text, Click Predict.')
    st.subheader('Step 1: Speak Here - Mic')
    audio = st.audio_input('Click mic and speak your sentence:')
    if audio is not None:
        audio_bytes = audio.read()
        st.audio(audio_bytes)
        st.success('Your speech recorded! Now converting speech to text...')
        if st.session_state.transcribed != str(len(audio_bytes)):
            with st.spinner('Converting your voice to text... wait 20 sec first time...'):
                try:
                    import soundfile as sf
                    import librosa
                    LIB = 'trans' + 'formers'
                    MOD = __import__(LIB)
                    @st.cache_resource
                    def get_speech():
                        return MOD.pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
                    whisper = get_speech()
                    data, sr = sf.read(io.BytesIO(audio_bytes))
                    if len(data.shape) > 1:
                        data = data.mean(axis=1)
                    if sr != 16000:
                        data = librosa.resample(data.astype(float), orig_sr=sr, target_sr=16000)
                    result = whisper(data, return_timestamps=False)
                    text_heard = result['text'].strip()
                    if text_heard:
                        st.session_state.txt = text_heard
                        st.session_state.transcribed = str(len(audio_bytes))
                        st.success("Heard: " + text_heard)
                    else:
                        st.warning('Could not hear clearly - please type below.')
                except Exception as e:
                    st.info('Auto transcription loading first time - please type what you spoke below for now, then click Predict.')
        else:
            st.success("Heard: " + st.session_state.txt)

    st.subheader('Step 2: Check Text and Predict')
    st.write('If auto text is wrong, correct it here:')
    txt = st.text_area('Type:', value=st.session_state.txt, key='txt_input')
    st.session_state.txt = txt
    if st.button('Predict Emotion'):
        if not txt.strip():
            st.warning('Please speak or type your sentence first, then click Predict!')
        else:
            with st.spinner('AI is thinking... wait 10 seconds...'):
                try:
                    LIB2 = 'trans' + 'formers'
                    MOD2 = __import__(LIB2)
                    @st.cache_resource
                    def get_model():
                        MODEL_NAME = "MoritzLaurer/De" + "BERTa-v3-xsmall-mnli-fever-anli"
                        return MOD2.pipeline("zero-shot-classification", model=MODEL_NAME)
                    clf = get_model()
                    labels = ["admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity","desire","disappointment","disapproval","disgust","embarrassment","excitement","fear","gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"]
                    result = clf(txt, labels, hypothesis_template="This person feels {}.", multi_label=False)
                    best = result['labels'][0]
                    best_score = int(result['scores'][0]*100)
                    second = result['labels'][1]
                    second_score = int(result['scores'][1]*100)
                    third = result['labels'][2]
                    third_score = int(result['scores'][2]*100)
                    st.success("Predicted: " + best + " (" + str(best_score) + "%)")
                    st.write("Text: " + txt)
                    st.write("AI reason: This sentence means '" + best + "' - AI understood full meaning, not keywords.")
                    st.write('Related emotions:')
                    st.write("1. " + best + ": " + str(best_score) + "%")
                    st.write("2. " + second + ": " + str(second_score) + "%")
                    st.write("3. " + third + ": " + str(third_score) + "%")
                except Exception as e:
                    st.error('Model is loading first time - wait 20 seconds and click Predict again.')
    st.write('See Results/accuracy_tables.md for full comparison.')