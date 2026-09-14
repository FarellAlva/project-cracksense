import json
import sys

nb_path = sys.argv[1]
cell_idx = int(sys.argv[2])
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
cell = nb['cells'][cell_idx]
print(f"=== Cell {cell_idx} [{cell['cell_type']}] ===")
print(''.join(cell['source']))
