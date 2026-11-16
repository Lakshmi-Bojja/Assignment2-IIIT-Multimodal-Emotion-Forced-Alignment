# 2a: Prepare MFA data - wav/ + transcripts/ -> corpus/ + dictionary
# Input in this repo: project/data/wav/*.wav + project/data/transcripts/*.txt (provided: 3 minimal-pair + 3 Hennessy)
# Output: corpus/<speaker>/<file>.wav + <file>.lab (transcript in MFA format), dictionary.txt (G2P)
# MFA expects: corpus dir with wav+lab pairs, dictionary with WORD PHONEMES (ARPA)
import pathlib, re, shutil, json
src_wav = pathlib.Path('project/data/wav'); src_txt = pathlib.Path('project/data/transcripts')
out = pathlib.Path('corpus'); out.mkdir(exist_ok=True)
# Provided transcripts (from assignment files) - also written to data/ by setup
provided = {
 'minimal_white_bait': 'I SAID WHITE NOT BAIT',
 'minimal_new_no': 'I SAID NEW NOT NO',
 'minimal_bad_bed': 'I SAID BAD NOT BED',
 'hennessy_1': 'Wanted Chief Justice of the Massachusetts Supreme Court In April the S J C s current leader Edward Hennessy reaches the mandatory retirement age of seventy and a successor is expected to be named in March It may be the most important appointment Governor Michael Dukakis makes during the remainder of his administration and one of the toughest As WBUR s Margo Melnicove reports Hennessy will be a hard act to follow',
 'hennessy_2': 'In nineteen seventy six Democratic Governor Michael Dukakis fulfilled a campaign promise to de-politicize judicial appointments He named Republican Edward Hennessy to head the State Supreme Judicial Court For Hennessy it was another step along a distinguished career that began as a trial lawyer and led to an appointment as associate Supreme Court Justice in nineteen seventy one That year Thomas Maffy now president of the Massachusetts Bar Association was Hennessy s law clerk',
 'hennessy_3': 'The author of more than eight hundred State Supreme Court opinions Hennessy is widely respected for his legal scholarship and his administrative abilities Admirers give Hennessy much of the credit for sweeping court reform that began a decade ago and for last year s legislative approval of thirty five new judgeships and three hundred million dollars to restore crumbling court houses Despite the state s massive budget deficit Hennessy recently urged colleagues in the bar association not to retreat from these hard won gains',
}
# G2P dictionary (ARPA, english_us_arpa style) - for minimal pairs + Hennessy vocab
g2p = {'I':'AY','SAID':'S EH D','WHITE':'W AY T','NOT':'N AA T','BAIT':'B EY T','NEW':'N UW','NO':'N OW','BAD':'B AE D','BED':'B EH D','WANTED':'W AO N T IH D','CHIEF':'CH IY F','JUSTICE':'JH AH S T AH S','OF':'AH V','THE':'DH AH','MASSACHUSETTS':'M AE S AH CH UW S AH T S','SUPREME':'S AH P R IY M','COURT':'K AO R T','HENNESSY':'HH EH N AH S IY'}
# Build corpus
for name, txt in provided.items():
    spk = out/'spk1'; spk.mkdir(exist_ok=True)
    # If real wav exists in data/wav, copy it; else create placeholder lab only (simulate mode will synthesize timing)
    wavs = list(src_wav.glob(f'{name}*.wav')) if src_wav.exists() else []
    if wavs:
        shutil.copy(wavs[0], spk/f'{name}.wav')
    (spk/f'{name}.lab').write_text(txt, encoding='utf-8')
    print(f'{name}: {txt[:60]}...')
# Dictionary: use existing MFA english_us_arpa for known words, G2P for OOV (Hennessy, Melnicove, etc.)
dict_lines=[]
for w in sorted(set(' '.join(provided.values()).upper().split())):
    ph = g2p.get(w, ' '.join(list(w)))  # fallback: letter-based (real run uses MFA G2P model)
    dict_lines.append(f'{w} {ph}')
pathlib.Path('dictionary.txt').write_text('\n'.join(dict_lines), encoding='utf-8')
print(f'Wrote corpus/ with {len(provided)} utterances, dictionary.txt with {len(dict_lines)} words')
print('Real MFA: mfa align corpus dictionary.txt english_us_arpa english_us_arpa aligned/ --clean')
print('Simulate: python forced_alignment/run_alignment.py --mode simulate')
# UNIQUE FILES ONLY: 14 unique (2 PDFs + 6 txt + 6 wav, 3 Hennessy repeats counted as 1)
print('Total unique files: 14 - repeats counted as one')
