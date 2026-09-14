import json
import sys

nb_path = sys.argv[1]
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']
print(f'Total cells: {len(cells)}')
for i, cell in enumerate(cells):
    src = ''.join(cell['source'])[:200].replace('\n', ' ')
    ctype = cell['cell_type']
    print(f'Cell {i} [{ctype}]: {src[:160]}')
