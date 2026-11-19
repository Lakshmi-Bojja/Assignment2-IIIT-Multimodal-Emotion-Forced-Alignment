# Assignment 2 - IIIT Multimodal Emotion + Forced Alignment
# Lakshmi Bojja | SLFI, LTRC, IIIT Hyderabad
# 2a: MFA TextGrid Viewer | 2b: Emotion from Text + Voice (AI predicts, no fixed sentences)
import os
import re
import streamlit as st
import pandas as pd

EMOJI = {
    "JOY": "😊",
    "SADNESS": "😢",
    "EXCITEMENT": "🤩",
    "CURIOSITY": "🤔",
    "SURPRISE": "😲",
    "DISAPPROVAL": "😒",
    "NEUTRAL": "😐",
    "ANGER": "😡",
    "FEAR": "😨",
    "LOVE": "❤️",
    "ADMIRATION": "😍",
    "AMUSEMENT": "😂",
    "CONFUSION": "😕",
    "DISAPPOINTMENT": "😞",
    "PRIDE": "😎",
    "GRATITUDE": "🙏",
    "NERVOUSNESS": "😬",
    "RELIEF": "😌",
    "DESIRE": "😍",
    "OPTIMISM": "🌈"
}

st.set_page_config(page_title="Assignment 2 - Emotion + Forced Alignment", layout="wide")
st.title("Assignment 2 - Multimodal Emotion + Forced Alignment")
st.caption("Lakshmi Bojja | SLFI, LTRC, IIIT Hyderabad | 231fa18166@gmail.com")

# ---------------- 2a : TextGrid Parser ----------------
def parse_textgrid(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    pattern = re.compile(r'intervals \[\d+\]:\s*xmin\s*=\s*([\d\.]+)\s*xmax\s*=\s*([\d\.]+)\s*text\s*=\s*"([^"]*)"', re.S)
    items = re.split(r'item \[\d+\]:', text)[1:]
    tiers = []
    for item in items:
        name_m = re.search(r'name\s*=\s*"([^"]+)"', item)
        ivs = pattern.findall(item)
        tiers.append((name_m.group(1) if name_m else "tier", ivs))
    return tiers

def tier_to_df(tier_intervals):
    rows = []
    for xmin, xmax, txt in tier_intervals:
        rows.append({"Start (s)": float(xmin), "End (s)": float(xmax), "Label": txt if txt else "(silence)", "Duration (s)": round(float(xmax)-float(xmin), 3)})
    return pd.DataFrame(rows)

# ---------------- 2b : Emotion Model ----------------
@st.cache_resource
def load_emotion():
    from transformers import pipeline
    clf = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device=-1)
    return clf

def predict_emotion(text):
    clf = load_emotion()
    out = clf(text)[0]
    out = sorted(out, key=lambda x: x["score"], reverse=True)
    top = out[0]
    return top["label"], top["score"], out[:5]

def show_result(label, score, top5):
    emoji = EMOJI.get(label.upper(), "✨")
    st.success(f"{emoji} Predicted Emotion: {label.upper()} - Confidence: {score*100:.1f}%")
    st.markdown(f"## {emoji} {label.upper()} — {score*100:.1f}%")
    st.write("Top 5 predictions:")
    df = pd.DataFrame([{"Emotion": x["label"], "Score": round(x["score"]*100, 2)} for x in top5])
    st.bar_chart(df.set_index("Emotion"))
    st.dataframe(df, use_container_width=True)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["2a - Forced Alignment (MFA + Praat)", "2b - Emotion from Text / Voice"])

with tab1:
    st.header("2a - Montreal Forced Aligner (MFA)")
    st.markdown("""
    **Name:** Lakshmi Bojja
    **Method:** Montreal Forced Aligner (MFA), acoustic model english_mfa, dictionary english_us_arpa
    **Corpus:** corpus/spk1 - 6 utterances (3 minimal-pair + 3 Hennessy)
    **Speaker:** spk1 (synthetic TTS voice, Windows System.Speech)
    **Command:** mfa align corpus english_us_arpa "C:/Users/Malleswari Bojja/Downloads/english_mfa/english_mfa" aligned --clean
    **Result:** 6/6 utterances aligned in 23.6 seconds. Output in aligned/spk1/ - 6 .TextGrid files.
    """)
    st.markdown("**Example - minimal_white_bait.TextGrid:** Word tier: \"\", \"i\", \"said\", \"white\", \"not\", \"bait\", \"\" | Phone tier: \"\", \"spn\", \"spn\", \"spn\", \"spn\", \"spn\", \"\"")
    st.markdown("**Observation:** Word boundaries are correct and match the .lab transcripts. Phone tier falls back to spn (spoken noise) because audio is synthetic TTS, not natural human speech. MFA's acoustic model was trained on human speech, so it cannot resolve phones in robotic TTS. Dictionary was not the cause - same spn result with custom dictionary.txt (144 words) and with MFA's own english_us_arpa dictionary.")
    st.markdown("**Fix for space in username:** MFA fails with [WinError 5] Access is denied when conda env is under C:/Users/Malleswari Bojja/ (space in name). Fix: conda create --prefix C:/mfa_env and conda activate C:/mfa_env, then run MFA from Admin VS Code.")
    st.subheader("🎧 Praat Manual Check")
    st.markdown("""
    **File checked:** minimal_white_bait.TextGrid + minimal_white_bait.wav | **Total duration:** 2.264172 seconds
    **Word tier:** "", "i", "said", "white", "not", "bait", "" | **Phone tier:** spn, spn, spn, spn
    **Observation:** Word boundaries align with waveform energy bursts. white and bait are correctly segmented. Phone tier shows spn for all intervals, confirming MFA falls back to spoken-noise for synthetic TTS audio. No skipped words, no timing offset.
    **Proof:** praat_check/praat_minimal_white_bait.png
    """)
    if os.path.exists("praat_check/praat_minimal_white_bait.png"):
        st.image("praat_check/praat_minimal_white_bait.png", caption="Praat - minimal_white_bait: Sound + TextGrid, words tier + phones tier", use_container_width=True)
    else:
        st.info("Praat photo: praat_check/praat_minimal_white_bait.png - push this file to GitHub to show it here.")
    st.subheader("📊 View Any TextGrid")
    base = "aligned/spk1"
    if os.path.exists(base):
        files = sorted([f for f in os.listdir(base) if f.endswith(".TextGrid")])
        if files:
            choice = st.selectbox("Choose utterance:", files)
            tiers = parse_textgrid(os.path.join(base, choice))
            for name, ivs in tiers:
                st.write(f"Tier: {name} - {len(ivs)} intervals")
                st.dataframe(tier_to_df(ivs), use_container_width=True)
            wav = os.path.join("corpus/spk1", choice.replace(".TextGrid", ".wav"))
            if os.path.exists(wav):
                st.audio(wav)
        else:
            st.warning("No .TextGrid files found in aligned/spk1/")
    else:
        st.warning("Folder aligned/spk1/ not found in repo. Push your 6 TextGrids to GitHub.")

with tab2:
    st.header("2b - Emotion Prediction (AI predicts from meaning)")
    st.write("Type ANY sentence - or speak / upload audio. No fixed sentences, no keywords. AI predicts from meaning.")
    text = st.text_area("Type your sentence here:", placeholder="e.g. I am so happy today, I got selected!", height=100)
    if st.button("Predict Emotion from Text", type="primary"):
        if text.strip() == "":
            st.warning("Please type a sentence first.")
        else:
            with st.spinner("AI is predicting..."):
                label, score, top5 = predict_emotion(text)
            show_result(label, score, top5)
    st.divider()
    st.subheader("🎤 Speak / Upload Voice")
    st.write("Record in Streamlit OR upload a .wav file. It will transcribe your voice, then predict emotion.")
    audio = st.audio_input("Record your voice:")
    uploaded = st.file_uploader("Or upload .wav / .mp3 file:", type=["wav", "mp3", "m4a", "ogg"])
    audio_file = audio if audio is not None else uploaded
    if audio_file is not None:
        st.audio(audio_file)
        if st.button("Transcribe + Predict from Voice"):
            with st.spinner("Transcribing voice..."):
                try:
                    from transformers import pipeline
                    import tempfile
                    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=-1)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getvalue())
                        tmp_path = tmp.name
                    result = asr(tmp_path)
                    transcribed = result["text"]
                    st.write(f"You said: {transcribed}")
                    label, score, top5 = predict_emotion(transcribed)
                    show_result(label, score, top5)
                except Exception as e:
                    st.error(f"Voice error: {e}. Try typing the same sentence in text box above.")
    st.info("Model: SamLowe/roberta-base-go_emotions (RoBERTa-base, GoEmotions). Top-1 label + Top-5 scores shown.")