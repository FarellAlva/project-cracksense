"""
Test TFLite + simple resize langsung tanpa Jupyter.
Untuk memastikan Cell 24 yang baru berfungsi benar.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

TFLITE_PATH = 'cracksense_ResNet50.tflite'
IMG_SIZE    = (224, 224)
CLASS_NAMES = ['Diagonal', 'Horizontal', 'Vertikal']
TEST_DIR    = './dataset/test'

# Load TFLite
print(f"Loading: {TFLITE_PATH}")
interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print(f"  Input  : {inp['shape']}  {inp['dtype']}")
print(f"  Output : {out['shape']}  {out['dtype']}")
print()


def predict_tflite(img_path, use_tta=True):
    img_raw    = tf.io.read_file(img_path)
    img        = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
    img        = tf.cast(img, tf.float32)
    img_r      = tf.image.resize(img, IMG_SIZE)   # simple resize

    if use_tta:
        variants = [
            img_r,
            tf.image.flip_left_right(img_r),
            tf.image.flip_up_down(img_r),
            tf.image.rot90(img_r, k=1),
        ]
    else:
        variants = [img_r]

    preds_all = []
    for v in variants:
        v_norm = tf.keras.applications.resnet.preprocess_input(v)
        v_np   = tf.expand_dims(v_norm, 0).numpy().astype(np.float32)
        interp.set_tensor(inp['index'], v_np)
        interp.invoke()
        p = interp.get_tensor(out['index'])[0]
        preds_all.append(p)

    return np.mean(preds_all, axis=0)


print("=" * 65)
print("HASIL PREDIKSI TFLITE (simple resize + TTA)")
print("=" * 65)

benar = 0
total = 0

for fname in sorted(os.listdir(TEST_DIR)):
    if not fname.endswith('.png'):
        continue

    fpath = os.path.join(TEST_DIR, fname)
    preds = predict_tflite(fpath, use_tta=True)
    pred_idx   = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_idx]
    conf       = preds[pred_idx] * 100

    # Ground truth dari nama file
    if 'horizontal' in fname.lower():
        gt = 'Horizontal'
    elif 'vertikal' in fname.lower():
        gt = 'Vertikal'
    elif 'diagonal' in fname.lower():
        gt = 'Diagonal'
    else:
        gt = 'Unknown'

    ok = pred_class == gt
    if ok:
        benar += 1
    total += 1

    probs = " | ".join([f"{CLASS_NAMES[i]}:{preds[i]*100:.0f}%" for i in range(3)])
    status = "OK " if ok else "ERR"
    print(f"  [{status}] {fname:30s}  GT:{gt:10s}  -> {pred_class:10s} ({conf:.1f}%)  [{probs}]")

print()
print(f"Akurasi: {benar}/{total} = {benar/total*100:.1f}%")
