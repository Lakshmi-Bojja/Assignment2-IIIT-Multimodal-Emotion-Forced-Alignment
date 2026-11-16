import json, pathlib
print(open('Results/text_results.json').read() if pathlib.Path('Results/text_results.json').exists() else 'Run train.py first')
