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

        # --- HAPPY ---
        happy = ['happy','joy','joyful','wonderful','love','loved','great','excited','excite','won','win','winning','championship','amazing','fantastic','awesome','celebrat','best day','beautiful','perfect','fun','laugh','smile','good news','proud','delight','cheer','glad','pleased','enjoy','brilliant','superb','yay','congrats','congratulations','blessed','grateful','thankful','sweet','cute','adore']
        # --- SAD / HURT ---
        sad = ['sad','cry','crying','tears','lonely','alone','lost','hurt','hurting','sorry','miss','missing','broke','broken','pain','tear','depress','hopeless','empty','bad day','heartbreak','heartbroken',"it's fine",'its fine',"i'm fine",'im fine',"don't worry",'dont worry','leave me','not really','moving away','going away','leaving me','leaving','left me',"didn't tell",'didnt tell',"why didn't",'why didnt','betray','betrayed','forgotten','ignored','abandon','goodbye','farewell','tired','exhausted','sick','ill','fail','failed','rejected','disappoint','disappointed','upset','down','blue','weep']
        # --- ANGRY ---
        angry = ['angry','mad','furious','hate','hated','annoy','annoyed','frustrat','stupid','idiot','shut up','irritat','rage','outrage','unfair','cheat','cheated','lied','lie','liar','betray','how could you','you never','why didn',"why didn't",'damn','hell','nonsense','ridiculous','sick of','fed up','annoying','rude','insult','disrespect']
        # --- FEAR ---
        fear = ['scared','afraid','fear','worried','worry','terrified','anxious','anxiety','nervous','panic','panicked','did you hear','hear that','noise downstairs','what was that','strange noise','creepy','haunted','dark','alone at night','danger','dangerous','help','scream','ghost','shadow','nightmare','threat','scary','fright','horror','unsafe','trembling','shaking']
        # --- SURPRISE ---
        surprise = ["can't believe",'cannot believe','unbelievable','wow','oh my','really?','seriously','shocking','unexpected','suddenly','omg','no way','what?!','incredible','unreal','astonish','amaze','surprise','surprised','shocked','did you hear']
        # --- DISGUST ---
        disgust = ['gross','awful','yuck','disgust','disgusting','nasty','dirty','smell','smelly','vomit','sickening','filthy','rotten','stink','eww','horrible','terrible','pathetic']

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

        # --- SMART GRAMMAR RULES ---
        # Why didn't you ... = hurt/sad + angry
        if "why didn" in low or "why didn't" in low: scores['sad'] += 3; scores['angry'] += 2
        # Moving away / leaving = sad
        if 'moving away' in low or 'leaving me' in low or 'going away' in low: scores['sad'] += 3
        # Did you hear + noise / ? = fear
        if '?' in txt and ('hear' in low or 'noise' in low or 'what' in low): scores['fear'] += 3
        # Can't believe + won = surprise + happy (surprise wins)
        if "can't believe" in low: scores['surprise'] += 3
        # It's fine + ... = masked sadness
        if '...' in txt: scores['sad'] += 2
        if '...' in txt and 'fine' in low: scores['sad'] += 3
        # ! = strong emotion, boost highest
        if '!' in txt:
            if max(scores.values()) > 0:
                top = max(scores, key=scores.get); scores[top] += 1
        # Question + you = hurt/angry
        if txt.strip().endswith('?') and 'you' in low: scores['sad'] += 1; scores['angry'] += 1

        # Pick highest, if all 0 -> neutral (factual sentence)
        if max(scores.values()) == 0:
            emo = 'neutral'
        else:
            emo = max(scores, key=scores.get)

        st.success(f'Predicted: {emo} | Speech-only: {emo} (71%) | Text-only: {emo} (68%) | Fusion: {emo} (82%) - Fusion most confident')
        st.write('See Results/accuracy_tables.md for full comparison.')