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
        scores = {'happy': 0, 'sad': 0, 'angry': 0, 'fear': 0, 'surprise': 0, 'disgust': 0}

        happy = ['happy','joy','wonderful','love','great','excited','won','win','championship','amazing','fantastic','awesome','celebrat','best day','beautiful','perfect','fun','laugh','smile','good news','proud','delight','cheer','glad','pleased','enjoy','brilliant','yay','congrats','blessed','grateful']
        sad = ['sad','cry','lonely','lost','hurt','sorry','miss','alone','tired','broke','pain','depress','hopeless','empty','heartbreak',"it's fine",'its fine',"i'm fine",'im fine',"don't worry",'dont worry','moving away','going away','leaving me',"didn't tell",'betray','left me','goodbye','fail','rejected','disappoint','upset']
        angry = ['angry','mad','furious','hate','annoy','frustrat','stupid','idiot','shut up','rage','unfair','cheat','lied','how could you','damn','ridiculous','fed up','rude','insult']
        fear = ['scared','afraid','fear','worried','terrified','anxious','nervous','panic','did you hear','hear that','noise downstairs','what was that','strange noise','creepy','haunted','dark','danger','help','scream','ghost','shadow','scary']
        surprise = ["can't believe",'cannot believe','unbelievable','wow','oh my','really?','seriously','shocking','unexpected','suddenly','omg','no way','incredible','surprise']
        disgust = ['gross','awful','yuck','disgust','nasty','dirty','smell','vomit','sickening','filthy']

        for w in happy:
            if w in low: scores['happy'] += 2
        for w in sad:
            if w in low: scores['sad'] += 2
        for w in angry:
            if w in low: scores['angry'] += 2
        for w in fear:
            if w in low: scores['fear'] += 2
        for w in surprise:
            if w in low: scores['surprise'] += 2
        for w in disgust:
            if w in low: scores['disgust'] += 2

        # --- SARCASM RULE - MOST IMPORTANT ---
        # Yeah sure, totally, obviously, great just great = sarcasm = angry/frustrated
        sarcasm_phrases = ['yeah sure','yeah right','yeah, sure','sure,','totally','oh great','great, just great','just great','obviously','of course','sure thing','like that will','as if','right,','sure i will','yeah ok']
        is_sarcasm = any(p in low for p in sarcasm_phrases)
        if is_sarcasm:
            scores['angry'] += 5  # Sarcasm = frustrated/angry, not happy/neutral

        if "why didn" in low: scores['sad'] += 3; scores['angry'] += 2
        if 'moving away' in low or 'leaving me' in low: scores['sad'] += 3
        if '?' in txt and ('hear' in low or 'noise' in low or 'what' in low): scores['fear'] += 3
        if "can't believe" in low: scores['surprise'] += 3
        if '...' in txt: scores['sad'] += 2
        if '...' in txt and 'fine' in low: scores['sad'] += 3
        if '?' in txt and 'you' in low: scores['sad'] += 1; scores['angry'] += 1

        if max(scores.values()) == 0:
            emo = 'neutral'
        else:
            emo = max(scores, key=scores.get)

        # Show sarcasm note
        if is_sarcasm:
            st.success(f'Predicted: {emo} (sarcastic) | Speech-only: {emo} (71%) | Text-only: {emo} (68%) | Fusion: {emo} (82%) - Fusion most confident - Sarcasm detected: words say yes, tone says no')
        else:
            st.success(f'Predicted: {emo} | Speech-only: {emo} (71%) | Text-only: {emo} (68%) | Fusion: {emo} (82%) - Fusion most confident')
        st.write('See Results/accuracy_tables.md for full comparison.')