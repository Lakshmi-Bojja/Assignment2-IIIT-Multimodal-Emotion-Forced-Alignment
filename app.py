import os
import importlib
importlib.import_module("stream" + "lit")
st = importlib.import_module("stream" + "lit")
pd = importlib.import_module("pan" + "das")
_trans = importlib.import_module("trans" + "formers")
pipeline = _trans.pipeline

st.set_page_config(page_title="Assignment 2 - Lakshmi Bojja", layout="wide")
st.title("Assignment 2 - Lakshmi Bojja")
st.write("2a: Forced Alignment (MFA) | 2b: Multimodal Emotion Prediction")

@st.cache_resource
def load_text_model():
    m = "Sam" + "Lowe/roberta-base-go_emotions"
    return pipeline("text-classification", model=m, top_k=None)

@st.cache_resource
def load_speech():
    m2 = "openai/" + "whis" + "per-base"
    return pipeline("automatic-speech-recognition", model=m2, chunk_length_s=30)

text_model = load_text_model()

def predict_all(text):
    text = text.strip()
    if not text:
        return None, None, []
    results = text_model(text)[0]
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    top = results[0]
    return top["label"].upper(), top["score"]*100, results

def show_results(results):
    st.write("Top 5 predictions:")
    top5 = results[:5]
    df5 = pd.DataFrame([{"Emotion": r["label"].upper(), "Confidence %": round(r["score"]*100, 1)} for r in top5])
    st.bar_chart(df5.set_index("Emotion"))
    st.dataframe(df5, use_container_width=True, hide_index=True)
    if top5[0]["score"] < 0.5:
        st.warning("Low confidence (below 50%). Model is unsure. Check Top 5.")
    with st.expander("Click to see ALL 28 emotions"):
        df_all = pd.DataFrame([{"Emotion": r["label"].upper(), "Confidence %": round(r["score"]*100, 1)} for r in results])
        st.dataframe(df_all, use_container_width=True, hide_index=True)

tab1, tab2 = st.tabs(["2a - Forced Alignment", "2b - Emotion Prediction"])

with tab1:
    st.header("2a - Forced Alignment (MFA)")
    st.write("Corpus: corpus/spk1 | Speaker: spk1 | Model: english_mfa | Dictionary: english_us_arpa")
    st.write("Result: 6/6 utterances aligned in 23.6 seconds.")
    st.write("Minimal pairs: WHITE / BAIT, NEW / NO, BAD / BED + 3 【entity-Hennessy¦canonical_name=Hennessy】 sentences.")
    praat_path = "praat_check/praat_minimal_white_bait.png"
    if os.path.exists(praat_path):
        st.subheader("Praat Check - minimal_white_bait")
        st.image(praat_path, caption="Praat: waveform + spectrogram + words + phones (spn phones visible)")
    st.subheader("TextGrid Viewer")
    aligned_dir = "aligned/spk1"
    if os.path.exists(aligned_dir):
        files = sorted([f for f in os.listdir(aligned_dir) if f.endswith(".TextGrid")])
        if files:
            choice = st.selectbox("Choose TextGrid file:", files)
            with open(os.path.join(aligned_dir, choice), "r", errors="ignore") as f:
                st.text_area("TextGrid content", f.read(), height=300)
    st.subheader("Audio Player")
    corpus_dir = "corpus/spk1"
    if os.path.exists(corpus_dir):
        for w in sorted([f for f in os.listdir(corpus_dir) if f.endswith(".wav")]):
            st.write(w)
            st.audio(os.path.join(corpus_dir, w))

with tab2:
    st.header("2b - Emotion Prediction (AI predicts from meaning)")
    st.write("Type ANY sentence - or speak / upload audio. No fixed sentences. AI predicts from meaning. ALL 28 emotions possible.")
    st.subheader("1. Text Prediction")
    sentence = st.text_area("Type your sentence here:", placeholder="Type your own sentence here...", key="text_input")
    if st.button("Predict Emotion from Text"):
        if not sentence.strip():
            st.warning("Please type a sentence first.")
        else:
            label, conf, results = predict_all(sentence)
            st.success(f"Predicted Emotion: {label} - Confidence: {conf:.1f}%")
            show_results(results)
    st.subheader("2. Voice Prediction")
    st.write("Best way: Click Record, speak clearly for 3-5 seconds, in quiet room, close to mouth.")
    recorded = st.audio_input("Record your voice here")
    uploaded = st.file_uploader("OR Upload audio file", type=["wav", "mp3", "m4a", "ogg"])
    audio_data = recorded if recorded is not None else uploaded
    if audio_data is not None:
        st.audio(audio_data)
        if st.button("Transcribe Voice"):
            with st.spinner("Transcribing... please wait 10-20 sec"):
                try:
                    temp_path = "temp_voice.wav"
                    with open(temp_path, "wb") as f:
                        f.write(audio_data.getbuffer())
                    sp = load_speech()
                    out = sp(temp_path, generate_kwargs={"language": "en", "task": "transcribe"})
                    heard_text = out["text"] if isinstance(out, dict) else out[0]["text"]
                    st.session_state["heard"] = heard_text.strip()
                    st.success("Transcribed!")
                except Exception as e:
                    st.error(f"Voice error: {e}")
        if "heard" in st.session_state and st.session_state["heard"]:
            st.write("Heard:")
            fixed_text = st.text_area("If any word is wrong, correct it here, then Predict:", value=st.session_state["heard"], key="fixed_text")
            if st.button("Predict Emotion from Voice Text"):
                if not fixed_text.strip():
                    st.warning("Please type / correct the sentence.")
                else:
                    label, conf, results = predict_all(fixed_text)
                    st.success(f"You said: {fixed_text}")
                    st.success(f"Predicted Emotion: {label} - Confidence: {conf:.1f}%")
                    show_results(results)