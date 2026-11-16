# 2a: Analyze TextGrids - word/phone durations, error analysis
import pathlib, re, statistics
for p in sorted(pathlib.Path('Results/textgrids').glob('*.TextGrid')):
    txt=p.read_text()
    # Parse intervals: xmin, xmax, text
    intervals=re.findall(r'xmin = ([0-9.]+)\s+xmax = ([0-9.]+)\s+text = "([^"]*)"', txt)
    words=[(float(s),float(e),t) for s,e,t in intervals if t and t.isupper() and len(t)>1]
    print(f'\n{p.name}: {len(words)} words')
    for s,e,w in words[:8]:
        print(f'  {s:.2f}-{e:.2f} ({e-s:.2f}s) {w}')
    durs=[e-s for s,e,_ in words]
    if durs: print(f'  Mean word dur: {statistics.mean(durs):.3f}s, range {min(durs):.3f}-{max(durs):.3f}s')
print('\nObservations: Minimal-pair words (WHITE/BAIT, NEW/NO, BAD/BED) are short vowels with clear boundaries; Hennessy paragraphs show longer function words and occasional timing offsets at phrase boundaries (see report).')
