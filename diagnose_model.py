"""
Diagnostik mendalam untuk model ResNet50 yang selalu prediksi Vertikal.
Cek: class distribution, model output, preprocessing, sample dari dataset asli.
"""
import os
import sys
import json
import numpy as np

print("="*70)
print("DIAGNOSTIK LENGKAP: ResNet50 Selalu Prediksi Vertikal")
print("="*70)

# ============================================================
# 1. Cek distribusi dataset
# ============================================================
print("\n[1] DISTRIBUSI DATASET")
print("-"*40)

DATASET_DIR = './dataset'
CLASS_NAMES = ['Diagonal', 'Horizontal', 'Vertikal']

for cls in CLASS_NAMES:
    cls_path = os.path.join(DATASET_DIR, cls)
    if os.path.exists(cls_path):
        files = [f for f in os.listdir(cls_path)
                 if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
        print(f"  {cls:15s}: {len(files):4d} gambar")
    else:
        print(f"  {cls:15s}: FOLDER TIDAK DITEMUKAN!")

# ============================================================
# 2. Load model dan cek output layer
# ============================================================
print("\n[2] CEK MODEL")
print("-"*40)

import tensorflow as tf

MODEL_PATH = 'cracksense_ResNet50.keras'
if not os.path.exists(MODEL_PATH):
    print(f"  Model tidak ditemukan: {MODEL_PATH}")
    sys.exit(1)

model = tf.keras.models.load_model(MODEL_PATH)
print(f"  Model loaded: {MODEL_PATH}")
print(f"  Input shape : {model.input_shape}")
print(f"  Output shape: {model.output_shape}")

# Cek layer terakhir
last_layer = model.layers[-1]
print(f"  Last layer  : {last_layer.name} ({last_layer.__class__.__name__})")
print(f"  Activation  : {getattr(last_layer, 'activation', 'N/A')}")

# ============================================================
# 3. Test dengan gambar dummy
# ============================================================
print("\n[3] TEST GAMBAR DUMMY (all zeros, all ones, all 0.5)")
print("-"*40)

for desc, val in [("All zeros", 0.0), ("All ones (255)", 255.0), ("Mid gray (128)", 128.0)]:
    dummy = tf.fill([1, 224, 224, 3], val)
    dummy = tf.keras.applications.resnet.preprocess_input(dummy)
    pred = model.predict(dummy, verbose=0)[0]
    pred_class = CLASS_NAMES[np.argmax(pred)]
    print(f"  {desc:20s}: {pred_class:10s} | {[f'{p*100:.1f}%' for p in pred]}")

# ============================================================
# 4. Test dengan gambar dari dataset asli (bukan test folder)
# ============================================================
print("\n[4] TEST SAMPEL DARI DATASET TRAINING")
print("-"*40)

def preprocess_for_inference(img_path, img_size=(224,224)):
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    # Resize langsung ke 224x224 (simple resize, tidak preserve aspect)
    img_simple = tf.image.resize(img, img_size)
    img_simple = tf.keras.applications.resnet.preprocess_input(img_simple)
    return tf.expand_dims(img_simple, 0)

correct = 0
total = 0
for cls_idx, cls in enumerate(CLASS_NAMES):
    cls_path = os.path.join(DATASET_DIR, cls)
    files = [f for f in os.listdir(cls_path)
             if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))][:5]  # 5 sampel per kelas
    
    for fname in files:
        fpath = os.path.join(cls_path, fname)
        img_batch = preprocess_for_inference(fpath)
        pred = model.predict(img_batch, verbose=0)[0]
        pred_idx = np.argmax(pred)
        pred_class = CLASS_NAMES[pred_idx]
        is_correct = pred_idx == cls_idx
        if is_correct:
            correct += 1
        total += 1
        status = "OK" if is_correct else "SALAH"
        print(f"  [{status}] {cls:10s}/{fname:30s} -> {pred_class} ({pred[pred_idx]*100:.1f}%)")

print(f"\n  Akurasi pada sampel training: {correct}/{total} = {correct/total*100:.1f}%")

# ============================================================
# 5. Test custom images (dari folder test) dengan resize sederhana
# ============================================================
print("\n[5] TEST CUSTOM IMAGES DENGAN RESIZE SEDERHANA (bukan preserve aspect)")
print("-"*40)

test_dir = './dataset/test'
test_files = sorted(os.listdir(test_dir))

for fname in test_files:
    fpath = os.path.join(test_dir, fname)
    img_batch = preprocess_for_inference(fpath)
    pred = model.predict(img_batch, verbose=0)[0]
    pred_idx = np.argmax(pred)
    pred_class = CLASS_NAMES[pred_idx]
    probs_str = " | ".join([f"{CLASS_NAMES[i]}:{pred[i]*100:.0f}%" for i in range(3)])
    
    # Deteksi ground truth dari nama file
    if 'horizontal' in fname.lower():
        gt = 'Horizontal'
    elif 'vertikal' in fname.lower():
        gt = 'Vertikal'
    elif 'diagonal' in fname.lower():
        gt = 'Diagonal'
    else:
        gt = 'Unknown'
    
    status = "OK" if pred_class == gt else "SALAH"
    print(f"  [{status}] {fname:30s} -> {pred_class:10s} | GT:{gt:10s} | {probs_str}")

# ============================================================
# 6. Cek probabilitas raw sebelum softmax
# ============================================================
print("\n[6] CEK APAKAH MODEL TERDEGRADASI (output collapsed)")
print("-"*40)

# Kumpulkan semua prediksi pada test set
all_preds = []
test_imgs = [os.path.join(test_dir, f) for f in sorted(os.listdir(test_dir))
             if f.endswith('.png')]

for fp in test_imgs:
    img_batch = preprocess_for_inference(fp)
    pred = model.predict(img_batch, verbose=0)[0]
    all_preds.append(pred)

all_preds = np.array(all_preds)
print(f"  Mean prediksi per kelas:")
for i, cls in enumerate(CLASS_NAMES):
    print(f"    {cls:15s}: mean={all_preds[:,i].mean():.3f}, std={all_preds[:,i].std():.3f}")

print(f"\n  Distribusi prediksi (kelas mana yang terpilih):")
pred_classes = np.argmax(all_preds, axis=1)
for i, cls in enumerate(CLASS_NAMES):
    n = (pred_classes == i).sum()
    print(f"    {cls:15s}: {n}/{len(test_imgs)} = {n/len(test_imgs)*100:.0f}%")

print()
print("="*70)
print("SELESAI DIAGNOSTIK")
print("="*70)
