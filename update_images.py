import json
import os

notebooks = [
    'crack_detection_ResNet50_tf_fixed.ipynb',
    'crack_detection_MobileNetV2_tf_fixed.ipynb',
    'crack_detection_MobileNetV3Large_tf_fixed.ipynb',
    'crack_detection_DenseNet121_fixed.ipynb'
]

replacement_target = "'./dataset/test/diagonal_2.png',\n]"
replacement_text = """'./dataset/test/diagonal_2.png',
    './dataset/test/diagonal_3.png',
    './dataset/test/diagonal_4.png',
    './dataset/test/diagonal_5.png',
]"""

for nb_path in notebooks:
    if os.path.exists(nb_path):
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        changed = False
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                src = ''.join(cell['source'])
                if 'TEST_IMAGE_PATHS' in src and 'diagonal_1.png' in src:
                    if 'diagonal_3.png' not in src:
                        new_src = src.replace(replacement_target, replacement_text)
                        
                        # Fallback for spacing variations
                        if new_src == src:
                            target2 = "'./dataset/test/diagonal_2.png',\\n]"
                            new_src = src.replace(target2, replacement_text)
                            
                        # Split while keeping newline characters correctly
                        if new_src != src:
                            lines = []
                            for line in new_src.splitlines(True):
                                lines.append(line)
                            cell['source'] = lines
                            changed = True

        if changed:
            with open(nb_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f'Added diagonal 3,4,5 to {nb_path}')
