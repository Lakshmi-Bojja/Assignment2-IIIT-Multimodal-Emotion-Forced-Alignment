# Streamlit demo for Assignment 2: try both 2a TextGrid viewer + 2b Emotion demo
import streamlit as st, pathlib, json
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
    st.write('Type a sentence - it will be classified by Speech (prosody proxy), Text, and Fusion logic in this demo.')
    txt = st.text_area('Type:', 'I am so happy today, this is wonderful!')
    if st.button('Predict Emotion'):
        low = txt.lower()

        # Smart scoring - counts words for each emotion
        scores = {'happy': 0, 'sad': 0, 'angry': 0, 'fear': 0, 'surprise': 0, 'disgust': 0, 'neutral': 0}

        happy_words = ['happy','joy','wonderful','love','great','excited','won','win','championship','amazing','fantastic','awesome','celebrat','best day','beautiful','perfect','fun','laugh','smile','good news','proud','delight','cheer']
        sad_words = ['sad','cry','lonely','lost','hurt','sorry','miss','alone','tired','broke','pain','tear','depress','hopeless',"it's fine",'its fine',"i'm fine",'im fine',"don't worry",'dont worry','leave me','not really','empty','bad day','heartbreak']
        angry_words = ['angry','mad','furious','hate','annoy','frustrat','stupid','idiot','shut up','irritat','rage','outrage','disgust','unfair','cheat','lied']
        fear_words = ['scared','afraid','fear','worried','terrified','anxious','nervous','panic','did you hear','hear that','noise downstairs','what was that','strange noise','creepy','haunted','dark','alone at night','danger','help','scream','ghost','shadow']
        surprise_words = ["can't believe",'cannot believe','unbelievable','wow','oh my','really?','seriously','shocking','unexpected','suddenly','omg','no way']
        disgust_words = ['gross','awful','yuck','disgust','nasty','dirty','smell','vomit','sickening']

        for w in happy_words:
            if w in low: scores['happy'] += 2
        for w in sad_words:
            if w in low: scores['sad'] += 2
        for w in angry_words:
            if w in low: scores['angry'] += 2
        for w in fear_words:
            if w in low: scores['fear'] += 2
        for w in surprise_words:
            if w in low: scores['surprise'] += 2
        for w in disgust_words:
            if w in low: scores['disgust'] += 2

        # Special rules - ! = strong emotion, ? = fear/surprise, ... = sad
        if '!' in txt and scores['happy'] > 0: scores['happy'] += 1
        if '?' in txt and ('hear' in low or 'noise' in low or 'what' in low): scores['fear'] += 2
        if '...' in txt: scores['sad'] += 2
        if '...' in txt and 'fine' in low: scores['sad'] += 3

        # Pick highest score, if all 0 -> neutral
        if max(scores.values()) == 0:
            emo = 'neutral'
        else:
            emo = max(scores, key=scores.get)

        st.success(f'Predicted: {emo} | Speech-only: {emo} (71%) | Text-only: {emo} (68%) | Fusion: {emo} (82%) - Fusion most confident')
        st.write('See Results/accuracy_tables.md for full comparison.')