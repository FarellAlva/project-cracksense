"""
Jalankan inferensi TFLite untuk semua 15 gambar drone di 4 model
dan tampilkan akurasi per model per kelas
"""
import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

TEST_DIR   = './dataset/test'
CLASS_NAMES = ['Diagonal', 'Horizontal', 'Vertikal']
IMG_SIZE   = (224, 224)

MODELS = [
    {"name": "DenseNet121",     "tflite": "cracksense_DenseNet121.tflite",     "preprocess": "densenet"},
    {"name": "MobileNetV2",     "tflite": "cracksense_MobileNetV2.tflite",     "preprocess": "mobilenet_v2"},
    {"name": "MobileNetV3Large","tflite": "cracksense_MobileNetV3Large.tflite","preprocess": "mobilenet_v3"},
    {"name": "ResNet50",        "tflite": "cracksense_ResNet50.tflite",        "preprocess": "resnet50"},
]

# Semua 15 gambar dengan ground truth
test_images = []
for cls in ['horizontal', 'vertikal', 'diagonal']:
    for i in range(1, 6):
        fname = f'{cls}_{i}.png'
        gt = cls.capitalize() if cls != 'vertikal' else 'Vertikal'
        # Normalize class name to match CLASS_NAMES
        for cn in CLASS_NAMES:
            if cn.lower() == cls.lower():
                gt = cn
                break
        test_images.append({'path': os.path.join(TEST_DIR, fname), 'gt': gt, 'fname': fname})

def preprocess(img_path, preproc_type):
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE)
    if preproc_type == 'densenet':
        img = tf.keras.applications.densenet.preprocess_input(img)
    elif preproc_type == 'mobilenet_v2':
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    elif preproc_type == 'mobilenet_v3':
        img = tf.keras.applications.mobilenet_v3.preprocess_input(img)
    elif preproc_type == 'resnet50':
        img = tf.keras.applications.resnet50.preprocess_input(img)
    return tf.expand_dims(img, 0).numpy().astype(np.float32)

print(f"{'Model':<20} {'Total':>6} {'Diagonal':>10} {'Horizontal':>12} {'Vertikal':>10} {'Accuracy':>10}")
print("-" * 75)

all_results = {}
for m in MODELS:
    interp = tf.lite.Interpreter(model_path=m['tflite'])
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()

    correct = {'Diagonal': 0, 'Horizontal': 0, 'Vertikal': 0}
    total_c = {'Diagonal': 0, 'Horizontal': 0, 'Vertikal': 0}

    for item in test_images:
        if not os.path.exists(item['path']):
            continue
        img_np = preprocess(item['path'], m['preprocess'])
        interp.set_tensor(inp[0]['index'], img_np)
        interp.invoke()
        preds = interp.get_tensor(out[0]['index'])[0]
        pred_cls = CLASS_NAMES[int(np.argmax(preds))]
        total_c[item['gt']] += 1
        if pred_cls == item['gt']:
            correct[item['gt']] += 1

    total_correct = sum(correct.values())
    total_all     = sum(total_c.values())
    acc = total_correct / total_all * 100

    print(f"{m['name']:<20} {total_all:>6} {correct['Diagonal']:>5}/{total_c['Diagonal']:<4} {correct['Horizontal']:>6}/{total_c['Horizontal']:<5} {correct['Vertikal']:>5}/{total_c['Vertikal']:<4} {acc:>8.1f}%")
    all_results[m['name']] = {
        'correct': correct,
        'total':   total_c,
        'acc':     acc
    }

print()
print("# Python dict untuk update_all_notebooks.py:")
print("DRONE_RESULTS = {")
for mn, r in all_results.items():
    h_ok = r['correct']['Horizontal']
    v_ok = r['correct']['Vertikal']
    d_ok = r['correct']['Diagonal']
    total_ok = h_ok + v_ok + d_ok
    acc_val  = r['acc']
    print(f'    "{mn}": {{"Horizontal": {h_ok}, "Vertikal": {v_ok}, "Diagonal": {d_ok}, "total": {total_ok}, "acc": {acc_val:.2f}}},')
print("}")
