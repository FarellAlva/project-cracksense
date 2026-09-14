import json

with open('crack_detection_ResNet50_tf_fixed.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== VERIFIKASI CELL 24 ===")
src24 = ''.join(nb['cells'][24]['source'])
for kw in ['preprocess_for_inference', 'tf.image.resize(img, IMG_SIZE)', 'Simple resize', 'use_tta']:
    status = "OK" if kw in src24 else "MISSING"
    print(f"  [{status}] {kw}")

print()
print("=== CELL 24 (50 baris pertama) ===")
lines = src24.split('\n')
for i, line in enumerate(lines[:50]):
    print(f"  {i+1:3d}: {line}")
