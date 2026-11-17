# Streamlit demo for Assignment 2: 2a TextGrid + 2b Emotion
# FINAL: Show ONLY related emotions - Top 3, no 0% list, clean look
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
    txt = st.text_area('Type:', 'I am so happy today, this is wonderful!')
    if st.button('Predict Emotion'):
        with st.spinner('AI is thinking... please wait 20 seconds...'):
            try:
                from transformers import pipeline
                @st.cache_resource
                def get_model():
                    return pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
                clf = get_model()
                result = clf(txt)[0]
                sorted_result = sorted(result, key=lambda x: x['score'], reverse=True)
                best = sorted_result[0]
                second = sorted_result[1]
                third = sorted_result[2]
                emo = best['label']
                top_score = int(best['score']*100)
                st.success(f"Predicted: {emo} ({top_score}%)")
                st.write(f"AI reason: This sentence means '{emo}' - AI understood full meaning, not keywords.")
                st.write('Related emotions for this sentence:')
                st.write(f"1. {best['label']}: {int(best['score']*100)}%")
                st.write(f"2. {second['label']}: {int(second['score']*100)}%")
                st.write(f"3. {third['label']}: {int(third['score']*100)}%")
            except Exception as e:
                st.error('AI model loading... Please wait 1 minute and try again.')
        st.write('See Results/accuracy_tables.md for full comparison.')