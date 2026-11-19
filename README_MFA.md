# Assignment 2a — Forced Alignment (MFA)

Name: Lakshmi Bojja
Method: Montreal Forced Aligner (MFA), acoustic model english_mfa, dictionary model english_us_arpa
Corpus: corpus/spk1 — 6 utterances (3 minimal-pair + 3 Hennessy)
Speaker: spk1 (synthetic TTS voice, Windows System.Speech)

Command:
mfa align corpus english_us_arpa "C:/Users/Malleswari Bojja/Downloads/english_mfa/english_mfa" aligned --clean

Result: 6/6 utterances aligned in 23.6 seconds. Output in aligned/spk1/ — 6 .TextGrid files.

Example — minimal_white_bait.TextGrid:
Word tier: "", "i", "said", "white", "not", "bait", ""
Phone tier: "", "spn", "spn", "spn", "spn", "spn", ""

Observation:
Word boundaries are correct and match the .lab transcripts.
Phone tier falls back to "spn" (spoken noise) because the audio is synthetic TTS,
not natural human speech. MFA's acoustic model was trained on human speech,
so it cannot resolve phones in robotic TTS. Dictionary was not the cause —
same spn result with custom dictionary.txt (144 words) and with MFA's own
english_us_arpa dictionary.

Fix for space in username:
MFA fails with [WinError 5] Access is denied when the conda env is under
C:/Users/Malleswari Bojja/ (space in name). Fix: conda create --prefix C:/mfa_env
and conda activate C:/mfa_env, then run MFA from Admin VS Code.

## Praat Manual Check

File checked: minimal_white_bait.TextGrid + minimal_white_bait.wav
Total duration: 2.264172 seconds

Word tier: "", "i", "said", "white", "not", "bait", ""
Phone tier: spn, spn, spn, spn

Observation: Word boundaries align with waveform energy bursts.
"white" and "bait" are correctly segmented. Phone tier shows
spn for all intervals, confirming MFA falls back to spoken-noise
for synthetic TTS audio. No skipped words, no timing offset.

Proof: praat_check/praat_minimal_white_bait.png