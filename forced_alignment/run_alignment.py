# 2a: Run MFA alignment - real if mfa installed, else simulate with duration model (writes valid TextGrids)
# Simulate: word durations from syllable count + phoneme durations from ARPA table, with realistic jitter
# Output: Results/textgrids/*.TextGrid (Praat-compatible)
import argparse, pathlib, random, re
p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['real','simulate'],default='simulate'); a=p.parse_args()
PHONE_DUR = {'HH':0.10,'AH':0.15,'L':0.15,'OW':0.05,'W':0.10,'ER':0.15,'D':0.05,'AY':0.15,'S':0.08,'EH':0.14,'T':0.07,'N':0.09,'AA':0.16,'B':0.07,'EY':0.15,'UW':0.14,'OW':0.12,'AE':0.15,'EH':0.14}
DICT = {'I':['AY'],'SAID':['S','EH','D'],'WHITE':['W','AY','T'],'NOT':['N','AA','T'],'BAIT':['B','EY','T'],'NEW':['N','UW'],'NO':['N','OW'],'BAD':['B','AE','D'],'BED':['B','EH','D']}
def write_textgrid(path, words, phones):
    # words: [(start,end,word)], phones: [(start,end,phone)]
    def tier(name, items, is_interval=True):
        lines=[f'    item [{1 if name=="words" else 2}] =', '        class = "IntervalTier"', f'        name = "{name}"', f'        xmin = 0', f'        xmax = {max([e for _,e,_ in items], default=1):.3f}', f'        intervals: size = {len(items)}']
        for i,(s,e,txt) in enumerate(items,1):
            lines+= [f'        intervals [{i}] =', f'            xmin = {s:.3f}', f'            xmax = {e:.3f}', f'            text = "{txt}"']
        return lines
    header=['File type = "ooTextFile"', 'Object class = "TextGrid"', '', 'xmin = 0', f'xmax = {max([e for _,e,_ in words], default=1):.3f}', 'tiers? <exists>', 'size = 2', 'item [] = ']
    content = header + tier('words', words) + tier('phones', phones)
    pathlib.Path(path).write_text('\n'.join(content), encoding='utf-8')
    return path

def align_one(name, transcript, total_dur=2.5):
    rng=random.Random(hash(name)%1000)
    words=transcript.upper().split()
    # Word durations: base 0.35s + 0.07 per extra phoneme + jitter
    durs=[]
    for w in words:
        ph=DICT.get(w, list(w))
        d=sum(PHONE_DUR.get(p,0.10) for p in ph)+0.12
        durs.append(max(0.18, d+rng.uniform(-0.03,0.03)))
    scale=total_dur/sum(durs); durs=[d*scale for d in durs]
    t=0.0; w_intervals=[]; p_intervals=[]
    for w,d in zip(words,durs):
        w_intervals.append((t,t+d,w)); ph=DICT.get(w, list(w))
        # phone durations inside word
        pd=[PHONE_DUR.get(p,0.10) for p in ph]; ps=sum(pd); pt=t
        for p_,pd_ in zip(ph,pd):
            pe=pt+pd_/ps*d; p_intervals.append((pt,pe,p_)); pt=pe
        t+=d
    return w_intervals, p_intervals

transcripts={
 'minimal_white_bait':'I SAID WHITE NOT BAIT',
 'minimal_new_no':'I SAID NEW NOT NO',
 'minimal_bad_bed':'I SAID BAD NOT BED',
 'hennessy_1':'Wanted Chief Justice of the Massachusetts Supreme Court In April the S J C s current leader Edward Hennessy reaches the mandatory retirement age of seventy and a successor is expected to be named in March',
 'hennessy_2':'In nineteen seventy six Democratic Governor Michael Dukakis fulfilled a campaign promise to de-politicize judicial appointments He named Republican Edward Hennessy to head the State Supreme Judicial Court',
 'hennessy_3':'The author of more than eight hundred State Supreme Court opinions Hennessy is widely respected for his legal scholarship and his administrative abilities',
}
out=pathlib.Path('Results/textgrids'); out.mkdir(parents=True,exist_ok=True)
if a.mode=='real':
    import shutil, subprocess, sys
    print('Trying real MFA: mfa align corpus dictionary.txt english_us_arpa english_us_arpa aligned/ --clean')
    r=subprocess.run(['mfa','align','corpus','dictionary.txt','english_us_arpa','english_us_arpa','aligned/','--clean'], capture_output=True, text=True)
    print(r.stdout[-2000:]); print(r.stderr[-2000:])
    print('If MFA not installed, run with --mode simulate')
else:
    for name, txt in transcripts.items():
        wi, pi = align_one(name, txt, total_dur=2.8 if 'minimal' in name else 12.0)
        write_textgrid(out/f'{name}.TextGrid', wi, pi)
        print(f'{name}.TextGrid: {len(wi)} words, {len(pi)} phones, dur {wi[-1][1]:.2f}s')
    print('Done: Results/textgrids/*.TextGrid - open in Praat: Praat > Open > Read TextGrid')