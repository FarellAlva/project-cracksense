import json

with open('crack_detection_ResNet50_tf_fixed.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ====================================================
# FIX 1: Cell 10 - Augmentasi Training (tambah flip/rotate)
# ====================================================

NEW_CELL10 = [
    "RESIZE_TO   = 256   # resize shortest side ke 256 \n",
    "CROP_SIZE   = 224   # final crop ke 224x224\n",
    "\n",
    "\n",
    "def resize_preserve_aspect(img):\n",
    "    \"\"\"Resize agar sisi terpendek = RESIZE_TO, pertahankan aspek rasio.\"\"\"\n",
    "    shape   = tf.shape(img)\n",
    "    h       = tf.cast(shape[0], tf.float32)\n",
    "    w       = tf.cast(shape[1], tf.float32)\n",
    "\n",
    "    scale   = tf.cast(RESIZE_TO, tf.float32) / tf.minimum(h, w)\n",
    "\n",
    "    new_h   = tf.cast(tf.round(h * scale), tf.int32)\n",
    "    new_w   = tf.cast(tf.round(w * scale), tf.int32)\n",
    "\n",
    "    new_h   = tf.maximum(new_h, RESIZE_TO)\n",
    "    new_w   = tf.maximum(new_w, RESIZE_TO)\n",
    "\n",
    "    img     = tf.image.resize(img, [new_h, new_w],\n",
    "                              method=tf.image.ResizeMethod.BILINEAR)\n",
    "    return img\n",
    "\n",
    "\n",
    "def center_crop(img):\n",
    "    shape  = tf.shape(img)\n",
    "    h, w   = shape[0], shape[1]\n",
    "    offset_h = (h - CROP_SIZE) // 2\n",
    "    offset_w = (w - CROP_SIZE) // 2\n",
    "    img    = tf.image.crop_to_bounding_box(img, offset_h, offset_w,\n",
    "                                           CROP_SIZE, CROP_SIZE)\n",
    "    return img\n",
    "\n",
    "def load_and_resize(path, label):\n",
    "    img   = tf.io.read_file(path)\n",
    "    img   = tf.image.decode_image(img, channels=3, expand_animations=False)\n",
    "    img   = tf.cast(img, tf.float32)          # [0, 255]\n",
    "    img   = resize_preserve_aspect(img)        # shortest side = 256\n",
    "    label = tf.one_hot(label, NUM_CLASSES)\n",
    "    return img, label\n",
    "\n",
    "def train_crop_and_augment(img, label):\n",
    "    # Random crop: posisi crop random di dalam area 256x256\n",
    "    img = tf.image.random_crop(img, size=[CROP_SIZE, CROP_SIZE, 3])\n",
    "\n",
    "    # ============================================================\n",
    "    # AUGMENTASI ORIENTASI - penting agar model tidak bias orientasi\n",
    "    # Tanpa ini, model cenderung memprediksi kelas yang paling\n",
    "    # banyak muncul dalam data training (mis. Vertikal).\n",
    "    # ============================================================\n",
    "\n",
    "    # Random flip horizontal (kiri-kanan)\n",
    "    img = tf.image.random_flip_left_right(img)\n",
    "\n",
    "    # Random flip vertikal (atas-bawah)\n",
    "    img = tf.image.random_flip_up_down(img)\n",
    "\n",
    "    # Random rotasi 90 derajat (0, 90, 180, 270)\n",
    "    # SANGAT PENTING untuk retakan diagonal/horizontal agar model\n",
    "    # belajar bahwa orientasi gambar bukan fitur yang relevan\n",
    "    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)\n",
    "    img = tf.image.rot90(img, k=k)\n",
    "\n",
    "    # Augmentasi warna ringan (setelah crop, masih di range [0,255])\n",
    "    img = tf.image.random_brightness(img, max_delta=0.2 * 255.0)\n",
    "    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)\n",
    "    img = tf.clip_by_value(img, 0.0, 255.0)\n",
    "    return img, label\n",
    "\n",
    "\n",
    "def val_test_crop(img, label):\n",
    "    img = center_crop(img)\n",
    "    return img, label\n",
    "\n",
    "\n",
    "def apply_normalize(img, label):\n",
    "    \"\"\"Terapkan preprocessing ResNet50 (mean subtraction BGR).\"\"\"\n",
    "    img = resnet_preprocess(img)\n",
    "    return img, label\n",
    "\n",
    "def make_dataset(paths, labels, is_training=False, shuffle=False):\n",
    "    ds = tf.data.Dataset.from_tensor_slices((paths, labels))\n",
    "    ds = ds.map(load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)\n",
    "\n",
    "    if is_training:\n",
    "        ds = ds.map(train_crop_and_augment, num_parallel_calls=tf.data.AUTOTUNE)\n",
    "    else:\n",
    "        ds = ds.map(val_test_crop, num_parallel_calls=tf.data.AUTOTUNE)\n",
    "\n",
    "    ds = ds.map(apply_normalize, num_parallel_calls=tf.data.AUTOTUNE)\n",
    "\n",
    "    if shuffle:\n",
    "        ds = ds.shuffle(buffer_size=500, seed=SEED)\n",
    "\n",
    "    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)\n",
    "    return ds\n",
    "\n",
    "\n",
    "train_ds = make_dataset(X_train, y_train, is_training=True,  shuffle=True)\n",
    "val_ds   = make_dataset(X_val,   y_val,   is_training=False, shuffle=False)\n",
    "test_ds  = make_dataset(X_test,  y_test,  is_training=False, shuffle=False)\n",
    "\n",
    "print(f'  Train batches      : {len(train_ds)}')\n",
    "print(f'  Validation batches : {len(val_ds)}')\n",
    "print(f'  Test batches       : {len(test_ds)}')\n",
    "print()\n",
    "print('  Augmentasi Training yang digunakan:')\n",
    "print('  - Random crop (256 -> 224)')\n",
    "print('  - Random flip horizontal')\n",
    "print('  - Random flip vertikal')\n",
    "print('  - Random rotasi 90 derajat (0/90/180/270)')\n",
    "print('  - Random brightness & contrast')"
]

# ====================================================
# FIX 2: Cell 24 - Custom image prediction + TTA
# ====================================================

NEW_CELL24 = [
    "TEST_IMAGE_PATHS = [\n",
    "    './dataset/test/horizontal_1.png',\n",
    "    './dataset/test/horizontal_2.png',\n",
    "    './dataset/test/horizontal_3.png',\n",
    "    './dataset/test/horizontal_4.png',\n",
    "    './dataset/test/horizontal_5.png',\n",
    "\n",
    "    './dataset/test/vertikal_1.png',\n",
    "    './dataset/test/vertikal_2.png',\n",
    "    './dataset/test/vertikal_3.png',\n",
    "    './dataset/test/vertikal_4.png',\n",
    "    './dataset/test/vertikal_5.png',\n",
    "\n",
    "    './dataset/test/diagonal_1.png',\n",
    "    './dataset/test/diagonal_2.png',\n",
    "]\n",
    "\n",
    "\n",
    "def tta_predict(img_tensor):\n",
    "    \"\"\"\n",
    "    Test-Time Augmentation (TTA).\n",
    "    \n",
    "    Rata-ratakan prediksi dari beberapa orientasi gambar.\n",
    "    Ini mengatasi bias model terhadap orientasi tertentu\n",
    "    saat model belum dilatih dengan augmentasi flip/rotate.\n",
    "    \"\"\"\n",
    "    img_cropped = center_crop(img_tensor)\n",
    "    variants = [\n",
    "        img_cropped,                                   # Original\n",
    "        tf.image.flip_left_right(img_cropped),         # Flip horizontal\n",
    "        tf.image.flip_up_down(img_cropped),            # Flip vertikal\n",
    "        tf.image.rot90(img_cropped, k=1),              # Rotasi 90\n",
    "        tf.image.rot90(img_cropped, k=3),              # Rotasi 270\n",
    "    ]\n",
    "    preds_all = []\n",
    "    for v in variants:\n",
    "        v_norm = resnet_preprocess(tf.cast(v, tf.float32))\n",
    "        p = model.predict(tf.expand_dims(v_norm, 0), verbose=0)[0]\n",
    "        preds_all.append(p)\n",
    "    return np.mean(preds_all, axis=0)\n",
    "\n",
    "\n",
    "def predict_custom_images(image_paths, use_tta=True):\n",
    "    \"\"\"\n",
    "    Prediksi gambar custom.\n",
    "    \n",
    "    Args:\n",
    "        image_paths: list path gambar\n",
    "        use_tta: gunakan Test-Time Augmentation untuk hasil lebih akurat\n",
    "    \"\"\"\n",
    "    results = []\n",
    "\n",
    "    for test_image_path in image_paths:\n",
    "        print(f'\\n=== Uji Gambar: {test_image_path} ===')\n",
    "\n",
    "        if os.path.exists(test_image_path):\n",
    "            # Load gambar dengan PIL untuk tampilan asli\n",
    "            pil_img = Image.open(test_image_path).convert('RGB')\n",
    "            img_w, img_h = pil_img.size\n",
    "\n",
    "            # Load dengan TensorFlow untuk prediksi\n",
    "            img_raw    = tf.io.read_file(test_image_path)\n",
    "            img_tensor = tf.image.decode_image(img_raw, channels=3, expand_animations=False)\n",
    "            img_tensor = tf.cast(img_tensor, tf.float32)\n",
    "\n",
    "            # PENTING: resize_preserve_aspect MEMPERTAHANKAN ASPEK RASIO\n",
    "            # Gambar horizontal tetap horizontal, vertikal tetap vertikal\n",
    "            img_resized = resize_preserve_aspect(img_tensor)\n",
    "\n",
    "            # Tampilkan gambar asli vs input model\n",
    "            fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n",
    "\n",
    "            axes[0].imshow(pil_img)\n",
    "            axes[0].set_title(f'Gambar Asli ({img_w}x{img_h})', fontweight='bold')\n",
    "            axes[0].axis('off')\n",
    "\n",
    "            img_cropped_display = center_crop(img_resized).numpy().astype(np.uint8)\n",
    "            axes[1].imshow(img_cropped_display)\n",
    "            axes[1].set_title('Input Model (224x224)', fontweight='bold')\n",
    "            axes[1].axis('off')\n",
    "\n",
    "            plt.suptitle(f'{os.path.basename(test_image_path)}', fontsize=12, fontweight='bold')\n",
    "            plt.tight_layout()\n",
    "            plt.show()\n",
    "\n",
    "            # Prediksi\n",
    "            if use_tta:\n",
    "                preds = tta_predict(img_resized)\n",
    "                metode = 'TTA (5 augmentasi dirata-rata)'\n",
    "            else:\n",
    "                img_norm = resnet_preprocess(center_crop(img_resized))\n",
    "                preds = model.predict(tf.expand_dims(img_norm, 0), verbose=0)[0]\n",
    "                metode = 'Single inference'\n",
    "\n",
    "            pred_idx   = np.argmax(preds)\n",
    "            confidence = preds[pred_idx] * 100\n",
    "            pred_class = CLASS_NAMES[pred_idx]\n",
    "\n",
    "            print(f'  Metode     : {metode}')\n",
    "            print(f'  Prediksi   : {pred_class}')\n",
    "            print(f'  Keyakinan  : {confidence:.2f}%')\n",
    "            for i, cls in enumerate(CLASS_NAMES):\n",
    "                bar = '|' * int(preds[i] * 30)\n",
    "                print(f'    {cls:12s}: {preds[i]*100:6.2f}% {bar}')\n",
    "\n",
    "            results.append({\n",
    "                'path': test_image_path,\n",
    "                'prediction': pred_class,\n",
    "                'confidence': float(confidence),\n",
    "                'probabilities': {cls: float(preds[i] * 100) for i, cls in enumerate(CLASS_NAMES)}\n",
    "            })\n",
    "        else:\n",
    "            print(f'  [!] File tidak ditemukan: {test_image_path}')\n",
    "            results.append({\n",
    "                'path': test_image_path,\n",
    "                'prediction': None,\n",
    "                'confidence': None,\n",
    "                'probabilities': None\n",
    "            })\n",
    "\n",
    "    return results\n",
    "\n",
    "\n",
    "# Prediksi dengan TTA\n",
    "print('=' * 60)\n",
    "print('INFERENSI GAMBAR CUSTOM + TEST-TIME AUGMENTATION (TTA)')\n",
    "print('=' * 60)\n",
    "custom_results = predict_custom_images(TEST_IMAGE_PATHS, use_tta=True)\n",
    "\n",
    "# Ringkasan\n",
    "print()\n",
    "print('=' * 60)\n",
    "print('RINGKASAN HASIL PREDIKSI')\n",
    "print('=' * 60)\n",
    "for r in custom_results:\n",
    "    if r['prediction'] is not None:\n",
    "        fname = os.path.basename(r['path'])\n",
    "        print(f'  {fname:30s} -> {r[\"prediction\"]:12s} ({r[\"confidence\"]:.1f}%)')\n",
    "print()\n",
    "print('CATATAN: Jika hasil masih salah setelah TTA, model perlu')\n",
    "print('di-retrain dengan augmentasi flip+rotate yang sudah diperbaiki')\n",
    "print('di Cell preprocessing di atas.')"
]

# Apply fixes
print("Menerapkan perbaikan ke notebook...")

cell10_src = ''.join(nb['cells'][10]['source'])
if 'resize_preserve_aspect' in cell10_src:
    nb['cells'][10]['source'] = NEW_CELL10
    nb['cells'][10]['outputs'] = []
    nb['cells'][10]['execution_count'] = None
    print("  OK: Cell 10 (preprocessing + augmentasi flip/rotate) diperbaiki")
else:
    print("  SKIP: Cell 10 tidak cocok")

cell24_src = ''.join(nb['cells'][24]['source'])
if 'predict_custom_images' in cell24_src:
    nb['cells'][24]['source'] = NEW_CELL24
    nb['cells'][24]['outputs'] = []
    nb['cells'][24]['execution_count'] = None
    print("  OK: Cell 24 (custom prediction + TTA) diperbaiki")
else:
    print("  SKIP: Cell 24 tidak cocok")

with open('crack_detection_ResNet50_tf_fixed.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print()
print("Selesai! Notebook berhasil diperbaiki.")
print()
print("="*60)
print("LANGKAH SELANJUTNYA:")
print("="*60)
print()
print("OPSI A - Langsung test (tanpa retrain):")
print("  Jalankan Cell 24 saja (TTA sudah diperbaiki)")
print("  Hasilnya mungkin masih agak kurang akurat")
print("  tapi lebih baik dari sebelumnya.")
print()
print("OPSI B - Retrain model (rekomendasi):")
print("  Jalankan ulang seluruh notebook dari awal")
print("  Model baru akan belajar augmentasi flip+rotate")
print("  sehingga lebih invariant terhadap orientasi gambar")
print("  -> Hasilnya akan jauh lebih akurat!")
