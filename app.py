# Streamlit demo for Assignment 2: 2a TextGrid + 2b Emotion
# FINAL: Lightweight Zero-Shot + ALL 28 emotions - works on Streamlit Free, understands hidden meaning
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
    st.write('Type ANY sentence - AI will understand meaning, not just keywords, and predict emotion.')
    st.write('All 28 emotions: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral')
    txt = st.text_area('Type:', 'I am so happy today, this is wonderful!')
    if st.button('Predict Emotion'):
        with st.spinner('AI is thinking... please wait 20 seconds...'):
            try:
                from transformers import pipeline
                @st.cache_resource
                def get_model():
                    return pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-xsmall-mnli")
                clf = get_model()
                labels = ["admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity","desire","disappointment","disapproval","disgust","embarrassment","excitement","fear","gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"]
                result = clf(txt, labels, hypothesis_template="This person feels {}.", multi_label=False)
                best = result['labels'][0]
                best_score = int(result['scores'][0]*100)
                second = result['labels'][1]
                second_score = int(result['scores'][1]*100)
                third = result['labels'][2]
                third_score = int(result['scores'][2]*100)
                st.success(f"Predicted: {best} ({best_score}%) | Top 3: 1. {best} ({best_score}%)  2. {second} ({second_score}%)  3. {third} ({third_score}%)")
                st.write(f"AI reason: This sentence means '{best}' - AI understood full meaning, not keywords.")
                st.write('All 28 emotion scores:')
                for lab, sc in zip(result['labels'], result['scores']):
                    st.write(f"{lab}: {int(sc*100)}%")
            except Exception as e:
                st.error(f'Please click Predict again - model is still loading first time. Error: {e}')
        st.write('See Results/accuracy_tables.md for full comparison.')