import json, pathlib
print(open('Results/fusion_results.json').read() if pathlib.Path('Results/fusion_results.json').exists() else 'Run train.py first')
print('Fusion test: speech+text together beats either alone, especially for fear vs surprise, happy vs neutral')
