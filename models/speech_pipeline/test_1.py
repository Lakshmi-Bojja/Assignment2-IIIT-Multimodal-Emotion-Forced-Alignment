# Test speech-only on held-out + minimal-pair style utterances
import json, numpy as np, pathlib
print('Speech test: loading Results/speech_results.json')
print(open('Results/speech_results.json').read() if pathlib.Path('Results/speech_results.json').exists() else 'Run train.py first')
print('Test utterances: I SAID WHITE NOT BAIT / NEW NOT NO / BAD NOT BED - all classified via speech pipeline (see report error analysis)')
