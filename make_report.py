# Make Assignment 2 Report PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = "Assignment2_Report_Lakshmi_Bojja.pdf"
c = canvas.Canvas(OUT, pagesize=A4)
W, H = A4
y = H - 50

def line(txt, gap=18, bold=False):
    global y
    if y < 60:
        c.showPage()
        y = H - 50
    c.setFont("Helvetica-Bold" if bold else "Helvetica", 12 if bold else 11)
    c.drawString(50, y, txt[:105])
    y -= gap

line("Assignment 2 - Lakshmi Bojja (231fa18166@gmail.com)", 22, True)
line("SLFI, LTRC, IIIT Hyderabad", 22)
line("", 10)
line("2a - Forced Alignment (MFA)", 20, True)
line("Corpus spk1: 6 utterances - 3 minimal pairs WHITE/BAIT, NEW/NO, BAD/BED", 16)
line("+ 3 Hennessy legal-scholarship sentences. MFA aligned 6/6 in 23.6 sec.", 16)
line("Example: minimal_white_bait.wav = 2.264172 sec", 16)
line("Word tier: i / said / white / not / bait. Phone tier: spn at start.", 16)
line("Hennessy example file: 8.0127 sec.", 16)
line("Figure: praat_check/praat_minimal_white_bait.png", 16)
line("", 10)
line("2b - Multimodal Emotion Recognition", 20, True)
line("Text: SamLowe/roberta-base-go_emotions (RoBERTa, 28 emotions).", 16)
line("Speech: Whisper-base transcribes speech, then same text model.", 16)
line("Fusion: Late fusion = average of text + speech scores.", 16)
line("Full model: MFCC + BiLSTM + Attention, Tokens + BiLSTM + Attention.", 16)
line("Cross-attention fusion + Classifier. See Results/accuracy_tables.md", 16)
line("", 10)
line("Results - TESS test n=560", 20, True)
line("Speech-only 0.8768, Text-only 0.8321, Fusion 0.8500", 16)
line("Fusion helps most for fear vs surprise, happy vs neutral.", 16)
line("", 10)
line("App tests (Streamlit)", 20, True)
line("I am so happy -> JOY 90.3%. I cleared final exams -> JOY 92.0%", 16)
line("My dog passed away -> SADNESS 98.4%. Get out of my room -> ANGER 80.1%", 16)
line("Please pass water bottle (voice) -> NEUTRAL 97.0%", 16)
line("Wait, how does this work? (voice) -> SURPRISE 62.4%", 16)
line("", 10)
line("Error cases", 20, True)
line("Sarcasm: Oh wonderful, another flat tire -> ADMIRATION 94.3% (fails).", 16)
line("Low confidence: I am not scared at all -> NEUTRAL 20.4%.", 16)
line("Why did you break my phone? -> CURIOSITY 27.0% (expected ANGER).", 16)
line("Someone is following me -> NEUTRAL 96.1% (expected FEAR).", 16)
line("", 10)
line("Links", 20, True)
line("GitHub: github.[https://github.com/Lakshmi-Bojja/Assignment2-IIIT-Multimodal-Emotion-Forced-Alignment]", 16)
line("Streamlit app: [https://assignment2-iiit-multimodal-emotion-forced-alignment-ccn6fhstu.streamlit.app/]", 16)
c.save()
print("Saved", OUT)