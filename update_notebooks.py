import json
import os

configs = [
    {
        'file': 'crack_detection_MobileNetV2_tf_fixed.ipynb',
        'tflite': 'cracksense_MobileNetV2.tflite',
        'preprocess_fn': 'mobilenetv2_preprocess'
    },
    {
        'file': 'crack_detection_MobileNetV3Large_tf_fixed.ipynb',
        'tflite': 'cracksense_MobileNetV3Large.tflite',
        'preprocess_fn': 'mobilenetv3_preprocess'
    },
    {
        'file': 'crack_detection_DenseNet121_fixed.ipynb',
        'tflite': 'cracksense_DenseNet121.tflite',
        'preprocess_fn': 'densenet_preprocess'
    }
]

# Cell 10 template
CELL_10_TEMPLATE = """RESIZE_TO   = 256   # resize shortest side ke 256 
CROP_SIZE   = 224   # final crop ke 224x224


def resize_preserve_aspect(img):
    \"\"\"Resize agar sisi terpendek menjadi RESIZE_TO, pertahankan aspek rasio.\"\"\"
    shape   = tf.shape(img)
    h       = tf.cast(shape[0], tf.float32)
    w       = tf.cast(shape[1], tf.float32)

    scale   = tf.cast(RESIZE_TO, tf.float32) / tf.minimum(h, w)

    new_h   = tf.cast(tf.round(h * scale), tf.int32)
    new_w   = tf.cast(tf.round(w * scale), tf.int32)

    new_h   = tf.maximum(new_h, RESIZE_TO)
    new_w   = tf.maximum(new_w, RESIZE_TO)

    img     = tf.image.resize(img, [new_h, new_w],
                              method=tf.image.ResizeMethod.BILINEAR)
    return img


def center_crop(img):
    shape  = tf.shape(img)
    h, w   = shape[0], shape[1]
    offset_h = (h - CROP_SIZE) // 2
    offset_w = (w - CROP_SIZE) // 2
    img    = tf.image.crop_to_bounding_box(img, offset_h, offset_w,
                                           CROP_SIZE, CROP_SIZE)
    return img

def load_and_resize(path, label):
    img   = tf.io.read_file(path)
    img   = tf.image.decode_image(img, channels=3, expand_animations=False)
    img   = tf.cast(img, tf.float32)          # [0, 255]
    img   = resize_preserve_aspect(img)        # shortest side = 256
    label = tf.one_hot(label, NUM_CLASSES)
    return img, label

def train_crop_and_augment(img, label):
    # Random crop: posisi crop random di dalam area 256x256
    img = tf.image.random_crop(img, size=[CROP_SIZE, CROP_SIZE, 3])

    # ============================================================
    # AUGMENTASI ORIENTASI - PENTING!
    # Mencegah model bias terhadap orientasi tertentu (mis. Vertikal)
    # ============================================================
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)

    # Augmentasi warna ringan
    img = tf.image.random_brightness(img, max_delta=0.2 * 255.0)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label


def val_test_crop(img, label):
    img = center_crop(img)
    return img, label


def apply_normalize(img, label):
    img = {PREPROCESS_FN}(img)
    return img, label

def make_dataset(paths, labels, is_training=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.map(train_crop_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(val_test_crop, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(apply_normalize, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=500, seed=SEED)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(X_train, y_train, is_training=True,  shuffle=True)
val_ds   = make_dataset(X_val,   y_val,   is_training=False, shuffle=False)
test_ds  = make_dataset(X_test,  y_test,  is_training=False, shuffle=False)

print(f'  Train batches      : {len(train_ds)}')
print(f'  Validation batches : {len(val_ds)}')
print(f'  Test batches       : {len(test_ds)}')
print()
print('  Augmentasi Training:')
print('  - Random crop (256 -> 224)')
print('  - Random flip horizontal & vertikal')
print('  - Random rotasi 90 derajat (0/90/180/270)')
print('  - Random brightness & contrast')
"""

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
        
    # Inject variables into templates
    c10 = CELL_10_TEMPLATE.replace('{PREPROCESS_FN}', cfg['preprocess_fn'])
    c24 = CELL_24_TEMPLATE.replace('{TFLITE_MODEL}', cfg['tflite']).replace('{PREPROCESS_FN}', cfg['preprocess_fn'])
    
    # Find the correct cells
    cell10_idx = -1
    cell24_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'def train_crop_and_augment' in src:
                cell10_idx = i
            if 'def predict_custom_images' in src or 'TEST_IMAGE_PATHS' in src:
                cell24_idx = i
                
    if cell10_idx != -1:
        # Notebook source formatting (split by newline and add \n)
        nb['cells'][cell10_idx]['source'] = [line + '\n' for line in c10.split('\n')]
        nb['cells'][cell10_idx]['outputs'] = []
        nb['cells'][cell10_idx]['execution_count'] = None
    if cell24_idx != -1:
        nb['cells'][cell24_idx]['source'] = [line + '\n' for line in c24.split('\n')]
        nb['cells'][cell24_idx]['outputs'] = []
        nb['cells'][cell24_idx]['execution_count'] = None
        
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"Updated {fpath} (Cell 10: {cell10_idx}, Cell 24: {cell24_idx})")
