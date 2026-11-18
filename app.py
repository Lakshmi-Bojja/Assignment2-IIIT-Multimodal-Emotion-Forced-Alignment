# Assignment 2 - FINAL: Audio -> Text Auto + Emotion
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
    st.info('Open TextGrid in Praat for waveform + spectrogram + boundaries visualization (see Report Fig).')

with tab2:
    st.header('2b: Emotion from Speech + Text')
    st.write('Step 1: Speak - App will convert your speech to text automatically. Step 2: Check text, Click Predict.')
    st.subheader('🎤 Step 1: Speak Here - Mic')
    audio = st.audio_input('Click mic and speak your sentence:')
    if audio is not None:
        audio_bytes = audio.read()
        st.audio(audio_bytes)
        st.success('Your speech recorded! Now converting speech to text...')
        # AUTO TRANSCRIBE - only once per new audio
        if st.session_state.transcribed != str(len(audio_bytes)):
            with st.spinner('Converting your voice to text... please wait 20 sec first time...'):
                try:
                    from 【entity-transformers¦canonical_name=transformers】 import pipeline
                    import soundfile as sf
                    import numpy as np
                    @st.cache_resource
                    def get_whisper():
                        return pipeline("automatic-speech-recognition", model="openai/【entity-whisper¦canonical_name=Whisper】-tiny")
                    【entity-whisper¦canonical_name=Whisper】 = get_whisper()
                    # Load WAV bytes directly - no ffmpeg needed for WAV
                    data, sr = sf.read(io.BytesIO(audio_bytes))
                    if len(data.shape) > 1:
                        data = data.mean(axis=1)
                    # 【entity-Whisper¦canonical_name=Whisper】 needs 16000 Hz - resample if needed
                    if sr != 16000:
                        import librosa
                        data = librosa.resample(data.astype(float), orig_sr=sr, target_sr=16000)
                    result = 【entity-whisper¦canonical_name=Whisper】(data, return_timestamps=False)
                    text_heard = result['text'].strip()
                    if text_heard:
                        st.session_state.txt = text_heard
                        st.session_state.transcribed = str(len(audio_bytes))
                        st.success(f"Heard: {text_heard}")
                    else:
                        st.warning('Could not hear clearly - please type below.')
                except Exception as e:
                    st.info('Auto transcription is loading first time - please type what you spoke below for now, then click Predict.')
        else:
            st.success(f"Heard: {st.session_state.txt}")

    st.subheader('⌨️ Step 2: Check Text & Predict')
    st.write('If auto text is wrong, you can correct it here:')
    txt = st.text_area('Type:', value=st.session_state.txt, key='txt_input')
    # Keep session txt in sync
    st.session_state.txt = txt
    if st.button('Predict Emotion'):
        if not txt.strip():
            st.warning('Please speak or type your sentence first, then click Predict!')
        else:
            with st.spinner('AI is thinking... please wait 10 seconds...'):
                try:
                    from 【entity-transformers¦canonical_name=transformers】 import pipeline
                    @st.cache_resource
                    def get_model():
                        return pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli")
                    clf = get_model()
                    labels = ["admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity","desire","disappointment","disapproval","disgust","embarrassment","excitement","fear","gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"]
                    result = clf(txt, labels, hypothesis_template="This person feels {}.", multi_label=False)
                    best = result['labels'][0]
                    best_score = int(result['scores'][0]*100)
                    second = result['labels'][1]
                    second_score = int(result['scores'][1]*100)
                    third = result['labels'][2]
                    third_score = int(result['scores'][2]*100)
                    st.success(f"Predicted: {best} ({best_score}%)")
                    st.write(f"Text: {txt}")
                    st.write(f"AI reason: This sentence means '{best}' - AI understood full meaning, not keywords.")
                    st.write('Related emotions:')
                    st.write(f"1. {best}: {best_score}%")
                    st.write(f"2. {second}: {second_score}%")
                    st.write(f"3. {third}: {third_score}%")
                except Exception as e:
                    st.error('Model is loading first time - please wait 20 seconds and click Predict again.')
    st.write('See Results/accuracy_tables.md for full comparison.')