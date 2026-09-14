import json
import os

configs = [
    {
        'file': 'crack_detection_MobileNetV2_tf_fixed.ipynb',
        'tflite': 'cracksense_MobileNetV2.tflite',
        'preprocess_fn': 'tf.keras.applications.mobilenet_v2.preprocess_input'
    },
    {
        'file': 'crack_detection_MobileNetV3Large_tf_fixed.ipynb',
        'tflite': 'cracksense_MobileNetV3Large.tflite',
        'preprocess_fn': 'tf.keras.applications.mobilenet_v3.preprocess_input'
    },
    {
        'file': 'crack_detection_DenseNet121_fixed.ipynb',
        'tflite': 'cracksense_DenseNet121.tflite',
        'preprocess_fn': 'tf.keras.applications.densenet.preprocess_input'
    }
]

# Cell 24 template
CELL_24_TEMPLATE = """# =============================================================
# INFERENSI GAMBAR CUSTOM MENGGUNAKAN TFLITE
# Tidak perlu compile ulang / load Keras model yang berat
# Preprocessing: simple resize ke 224x224 (TERBUKTI BENAR)
# =============================================================

TFLITE_PATH  = '{TFLITE_MODEL}'
IMG_SIZE     = (224, 224)
CLASS_NAMES  = ['Diagonal', 'Horizontal', 'Vertikal']

TEST_IMAGE_PATHS = [
    './dataset/test/horizontal_1.png',
    './dataset/test/horizontal_2.png',
    './dataset/test/horizontal_3.png',
    './dataset/test/horizontal_4.png',
    './dataset/test/horizontal_5.png',

    './dataset/test/vertikal_1.png',
    './dataset/test/vertikal_2.png',
    './dataset/test/vertikal_3.png',
    './dataset/test/vertikal_4.png',
    './dataset/test/vertikal_5.png',

    './dataset/test/diagonal_1.png',
    './dataset/test/diagonal_2.png',
]

# ----------------------------------------------------------
# Load TFLite interpreter
# ----------------------------------------------------------
print(f'Loading TFLite model: {{TFLITE_PATH}}')
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f'  Input  : shape={{input_details[0]["shape"]}}, dtype={{input_details[0]["dtype"]}}')
print(f'  Output : shape={{output_details[0]["shape"]}}, dtype={{output_details[0]["dtype"]}}')
print()

# ----------------------------------------------------------
# Fungsi preprocessing: simple resize ke 224x224
# PENTING: jangan pakai center_crop! (penyebab bug Vertikal semua)
# ----------------------------------------------------------
def preprocess_image_tflite(img_path):
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    
    # Simple resize langsung ke 224x224
    img = tf.image.resize(img, IMG_SIZE)
    
    # Preprocessing model-specific
    img = {PREPROCESS_FN}(img)
    
    img = tf.expand_dims(img, axis=0)
    return img.numpy().astype(np.float32)

# ----------------------------------------------------------
# Fungsi prediksi TFLite (dengan TTA opsional)
# ----------------------------------------------------------
def tflite_predict_single(img_np):
    interpreter.set_tensor(input_details[0]['index'], img_np)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

def tflite_predict_with_tta(img_path):
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    img_resized = tf.image.resize(img, IMG_SIZE)

    variants = [
        img_resized,
        tf.image.flip_left_right(img_resized),
        tf.image.flip_up_down(img_resized),
        tf.image.rot90(img_resized, k=1),
    ]

    preds_all = []
    for v in variants:
        v_norm = {PREPROCESS_FN}(v)
        v_np   = tf.expand_dims(v_norm, 0).numpy().astype(np.float32)
        p = tflite_predict_single(v_np)
        preds_all.append(p)

    return np.mean(preds_all, axis=0)

# ----------------------------------------------------------
# Fungsi utama: prediksi gambar custom
# ----------------------------------------------------------
def predict_custom_images_tflite(image_paths, use_tta=True):
    results = []
    for test_image_path in image_paths:
        print(f'\\n=== Uji Gambar: {{test_image_path}} ===')

        if not os.path.exists(test_image_path):
            print(f'  [!] File tidak ditemukan!')
            results.append({{'path': test_image_path, 'prediction': None, 'confidence': None, 'probabilities': None}})
            continue

        pil_img = Image.open(test_image_path).convert('RGB')
        img_w, img_h = pil_img.size
        img_display = np.array(pil_img.resize(IMG_SIZE[::-1]))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(pil_img)
        axes[0].set_title(f'Gambar Asli ({{img_w}}x{{img_h}})', fontweight='bold')
        axes[0].axis('off')
        axes[1].imshow(img_display)
        axes[1].set_title(f'Input Model (224x224 resize)', fontweight='bold')
        axes[1].axis('off')
        plt.suptitle(os.path.basename(test_image_path), fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

        if use_tta:
            preds  = tflite_predict_with_tta(test_image_path)
            metode = 'TFLite + TTA (4 varian orientasi)'
        else:
            img_np = preprocess_image_tflite(test_image_path)
            preds  = tflite_predict_single(img_np)
            metode = 'TFLite single inference'

        pred_idx   = int(np.argmax(preds))
        confidence = float(preds[pred_idx] * 100)
        pred_class = CLASS_NAMES[pred_idx]

        print(f'  Backend    : TFLite ({{TFLITE_PATH}})')
        print(f'  Metode     : {{metode}}')
        print(f'  Prediksi   : {{pred_class}}')
        print(f'  Keyakinan  : {{confidence:.2f}}%')
        for i, cls in enumerate(CLASS_NAMES):
            bar = '|' * int(preds[i] * 40)
            print(f'    {{cls:12s}}: {{preds[i]*100:6.2f}}% {{bar}}')

        results.append({{
            'path':          test_image_path,
            'prediction':    pred_class,
            'confidence':    confidence,
            'probabilities': {{cls: float(preds[i]*100) for i, cls in enumerate(CLASS_NAMES)}}
        }})

    return results

print('=' * 60)
print('INFERENSI TFLITE - GAMBAR CUSTOM')
print('=' * 60)
print(f'Model : {{TFLITE_PATH}}')
print()

custom_results = predict_custom_images_tflite(TEST_IMAGE_PATHS, use_tta=True)

print()
print('=' * 60)
print('RINGKASAN HASIL PREDIKSI')
print('=' * 60)
for r in custom_results:
    if r['prediction'] is not None:
        fname = os.path.basename(r['path'])
        print(f'  {{fname:30s}} -> {{r["prediction"]:12s}} ({{r["confidence"]:.1f}}%)')
"""

for cfg in configs:
    fpath = cfg['file']
    if not os.path.exists(fpath):
        print(f"File not found: {fpath}")
        continue
    
    with open(fpath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    c24 = CELL_24_TEMPLATE.replace('{TFLITE_MODEL}', cfg['tflite']).replace('{PREPROCESS_FN}', cfg['preprocess_fn'])
    
    cell24_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'TEST_IMAGE_PATH' in src or 'predict_custom_images' in src:
                cell24_idx = i
                
    if cell24_idx != -1:
        nb['cells'][cell24_idx]['source'] = [line + '\n' for line in c24.split('\n')]
        nb['cells'][cell24_idx]['outputs'] = []
        nb['cells'][cell24_idx]['execution_count'] = None
        
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Updated {fpath} (Cell 24: {cell24_idx})")
    else:
        print(f"Could not find cell 24 for {fpath}")
