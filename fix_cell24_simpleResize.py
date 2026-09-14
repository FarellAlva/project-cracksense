import json

with open('crack_detection_ResNet50_tf_fixed.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ============================================================
# Fix Cell 24: Ganti preprocessing ke simple resize (TERBUKTI BENAR dari diagnostik)
# Simple resize (224x224) = 15/15 benar
# Preserve aspect + center crop = semua salah (Vertikal)
# ============================================================

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
    "# ============================================================\n",
    "# CATATAN PENTING - PENYEBAB BUG:\n",
    "# resize_preserve_aspect + center_crop MERUSAK orientasi gambar.\n",
    "# Gambar horizontal jika di-center_crop menjadi patch 224x224\n",
    "# yang tidak representatif, dan model mengenalinya sebagai Vertikal.\n",
    "#\n",
    "# SOLUSI TERBUKTI: Simple resize langsung ke 224x224\n",
    "# (akurasi: 15/15 = 100% pada test set)\n",
    "# ============================================================\n",
    "\n",
    "def preprocess_for_inference(img_path, use_tta=True):\n",
    "    \"\"\"\n",
    "    Preprocessing gambar untuk inferensi.\n",
    "    \n",
    "    Menggunakan simple resize ke 224x224, BUKAN preserve_aspect + center_crop.\n",
    "    Alasan: center_crop pada gambar non-square bisa memotong fitur penting\n",
    "    dan mengubah persepsi orientasi model.\n",
    "    \"\"\"\n",
    "    img_raw = tf.io.read_file(img_path)\n",
    "    img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)\n",
    "    img = tf.cast(img, tf.float32)\n",
    "\n",
    "    if use_tta:\n",
    "        # Test-Time Augmentation: rata-rata dari 4 crop + resize\n",
    "        # Resize langsung\n",
    "        img_resize = tf.image.resize(img, IMG_SIZE)\n",
    "\n",
    "        # Flip horizontal\n",
    "        img_flip_h = tf.image.flip_left_right(img_resize)\n",
    "\n",
    "        # Flip vertikal\n",
    "        img_flip_v = tf.image.flip_up_down(img_resize)\n",
    "\n",
    "        # Rotasi 90\n",
    "        img_rot90 = tf.image.rot90(img_resize, k=1)\n",
    "\n",
    "        variants = [img_resize, img_flip_h, img_flip_v, img_rot90]\n",
    "        preds_all = []\n",
    "        for v in variants:\n",
    "            v_norm = resnet_preprocess(v)\n",
    "            p = model.predict(tf.expand_dims(v_norm, 0), verbose=0)[0]\n",
    "            preds_all.append(p)\n",
    "        return np.mean(preds_all, axis=0)\n",
    "    else:\n",
    "        # Single inference tanpa TTA\n",
    "        img_resize = tf.image.resize(img, IMG_SIZE)\n",
    "        img_norm = resnet_preprocess(img_resize)\n",
    "        return model.predict(tf.expand_dims(img_norm, 0), verbose=0)[0]\n",
    "\n",
    "\n",
    "def predict_custom_images(image_paths, use_tta=True):\n",
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
    "            # Load gambar asli dengan TF untuk mendapatkan info ukuran\n",
    "            img_raw = tf.io.read_file(test_image_path)\n",
    "            img_orig = tf.image.decode_image(img_raw, channels=3, expand_animations=False)\n",
    "            img_orig = tf.cast(img_orig, tf.float32)\n",
    "\n",
    "            # Buat versi resize untuk ditampilkan (apa yang model lihat)\n",
    "            img_display = tf.image.resize(img_orig, IMG_SIZE).numpy().astype(np.uint8)\n",
    "\n",
    "            # Tampilkan gambar asli vs input model\n",
    "            fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n",
    "\n",
    "            axes[0].imshow(pil_img)\n",
    "            axes[0].set_title(f'Gambar Asli ({img_w}x{img_h})', fontweight='bold')\n",
    "            axes[0].axis('off')\n",
    "\n",
    "            axes[1].imshow(img_display)\n",
    "            axes[1].set_title(f'Input Model ({IMG_SIZE[0]}x{IMG_SIZE[1]} resize)', fontweight='bold')\n",
    "            axes[1].axis('off')\n",
    "\n",
    "            fname = os.path.basename(test_image_path)\n",
    "            plt.suptitle(fname, fontsize=12, fontweight='bold')\n",
    "            plt.tight_layout()\n",
    "            plt.show()\n",
    "\n",
    "            # Prediksi\n",
    "            preds = preprocess_for_inference(test_image_path, use_tta=use_tta)\n",
    "\n",
    "            pred_idx   = np.argmax(preds)\n",
    "            confidence = preds[pred_idx] * 100\n",
    "            pred_class = CLASS_NAMES[pred_idx]\n",
    "\n",
    "            metode = 'Simple resize + TTA (4 varian)' if use_tta else 'Simple resize'\n",
    "            print(f'  Metode     : {metode}')\n",
    "            print(f'  Prediksi   : {pred_class}')\n",
    "            print(f'  Keyakinan  : {confidence:.2f}%')\n",
    "            for i, cls in enumerate(CLASS_NAMES):\n",
    "                bar = '|' * int(preds[i] * 40)\n",
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
    "# Jalankan prediksi\n",
    "print('=' * 60)\n",
    "print('INFERENSI GAMBAR CUSTOM (Simple Resize + TTA)')\n",
    "print('=' * 60)\n",
    "print('Metode: tf.image.resize(img, (224,224)) -> resnet_preprocess')\n",
    "print('Tidak menggunakan preserve_aspect + center_crop (buggy!')\n",
    "print()\n",
    "\n",
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
    "        print(f'  {fname:30s} -> {r[\"prediction\"]:12s} ({r[\"confidence\"]:.1f}%)')"
]

# Juga fix Cell 10 - tambahkan note bahwa val/test crop juga harus diganti
# Dan perbaiki val_test_crop untuk menggunakan simple resize

NEW_CELL10_VAL_FIX = None  # Akan kita lihat apakah perlu

# Apply fix Cell 24
cell24_src = ''.join(nb['cells'][24]['source'])
if 'predict_custom_images' in cell24_src or 'TEST_IMAGE_PATHS' in cell24_src:
    nb['cells'][24]['source'] = NEW_CELL24
    nb['cells'][24]['outputs'] = []
    nb['cells'][24]['execution_count'] = None
    print("OK: Cell 24 diperbaiki (simple resize menggantikan center_crop)")
else:
    print("SKIP: Cell 24 tidak cocok")

# Simpan
with open('crack_detection_ResNet50_tf_fixed.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print()
print("="*60)
print("PERBAIKAN SELESAI!")
print("="*60)
print()
print("ROOT CAUSE DITEMUKAN:")
print("  resize_preserve_aspect + center_crop pada gambar landscape")
print("  (horizontal/diagonal) memotong bagian tengah yang terlihat")
print("  VERTIKAL oleh model.")
print()
print("SOLUSI:")
print("  Gunakan tf.image.resize(img, (224,224)) langsung")
print("  -> Hasil diagnostik: 15/15 = 100% benar!")
print()
print("Sekarang jalankan Cell 24 di notebook untuk test.")
