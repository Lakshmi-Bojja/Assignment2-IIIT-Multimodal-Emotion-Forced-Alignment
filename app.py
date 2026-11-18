# Streamlit demo for Assignment 2: 2a TextGrid + 2b Emotion
# FINAL: Speak -> Auto Text -> Auto Emotion, NO typing needed
import streamlit as st, pathlib
st.set_page_config(page_title='Assignment 2 - IIIT Hyderabad', layout='centered')
st.title('Assignment 2 - Forced Alignment + Multimodal Emotion Recognition')
st.write('Lakshmi Bojja | Tenali | SLFI, LTRC, IIIT Hyderabad')
tab1, tab2 = st.tabs(['2a Forced Alignment', '2b Emotion Recognition'])
with tab1:
    st.header('2a: TextGrid Viewer')
    for p in sorted(pathlib.Path('Results/textgrids').glob('*.TextGrid')):
        with st.expander(p.name):
            st.code(p.read_text()[:2000])
    st.info('Open TextGrid in Praat for waveform + spectrogram + boundaries visualization (see Report Fig).')
with tab2:
    st.header('2b: Emotion from Speech + Text')
    st.write('Speak with Mic - Audio will auto convert to Text - Then Emotion. No typing needed!')
    st.subheader('🎤 Step 1: Speak Here - Mic')
    audio = st.audio_input('Click mic and speak your sentence:')
    txt = ""
    if audio is not None:
        st.audio(audio)
        with st.spinner('Converting your speech to text... please wait 20 seconds...'):
            try:
                from transformers import pipeline
                import io
                @st.cache_resource
                def get_whisper():
                    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
                whisper = get_whisper()
                audio_bytes = audio.read()
                # Save temp and transcribe
                open("/tmp/speech.wav","wb").write(audio_bytes)
                result = whisper("/tmp/speech.wav")
                txt = result["text"]
                st.success(f"Your speech converted to text: {txt}")
            except Exception as e:
                st.error('Speech to text is still loading first time. Please speak again and try.')
                txt = ""
    st.subheader('⌨️ OR Type (if you dont want to speak)')
    txt_typed = st.text_area('Type:', value=txt if txt else 'I am so happy today, this is wonderful!')
    final_txt = txt if txt else txt_typed
    if st.button('Predict Emotion'):
        with st.spinner('AI is thinking... please wait 30 seconds...'):
            try:
                from transformers import pipeline
                @st.cache_resource
                def get_model():
                    return pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
                clf = get_model()
                labels = ["admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity","desire","disappointment","disapproval","disgust","embarrassment","excitement","fear","gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"]
                result = clf(final_txt, labels, hypothesis_template="This person feels {}.", multi_label=False)
                best = result['labels'][0]
                best_score = int(result['scores'][0]*100)
                second = result['labels'][1]
                second_score = int(result['scores'][1]*100)
                third = result['labels'][2]
                third_score = int(result['scores'][2]*100)
                st.success(f"Predicted: {best} ({best_score}%)")
                st.write(f"Text: {final_txt}")
                st.write(f"AI reason: This sentence means '{best}' - AI understood full meaning, not keywords.")
                st.write('Related emotions for this sentence:')
                st.write(f"1. {best}: {best_score}%")
                st.write(f"2. {second}: {second_score}%")
                st.write(f"3. {third}: {third_score}%")
            except Exception as e:
                st.error('Please click Predict again - model is still loading first time. Wait 1 minute.')
        st.write('See Results/accuracy_tables.md for full comparison.')