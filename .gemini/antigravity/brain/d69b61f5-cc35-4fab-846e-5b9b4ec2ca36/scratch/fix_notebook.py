import json
import re

file_path = r'd:\Studia\OneDrive - Politechnika Wroclawska\II_stopien\1_sem\Glebokie_sieci\Laby\GlebokieSieciNeuronowe\lab06.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changed = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'torchvision.datasets.CIFAR10' in ''.join(cell['source']):
        source_str = ''.join(cell['source'])
        # Check if torchvision is imported as a module, not just transforms
        # Patterns: "import torchvision" followed by newline or space, but not "."
        if not re.search(r'\bimport torchvision\b(?!\.transforms)', source_str):
            new_source = []
            added = False
            for line in cell['source']:
                new_source.append(line)
                if 'import torch' in line and 'import torch.' not in line and not added:
                    new_source.append('import torchvision\n')
                    added = True
            if not added:
                new_source.insert(0, 'import torchvision\n')
            cell['source'] = new_source
            changed = True
            print("Fixed torchvision import in a cell.")

if changed:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
else:
    print("No changes made.")
