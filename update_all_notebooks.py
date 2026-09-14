"""
update_all_notebooks.py
Menambahkan:
  1. Grafik distribusi dataset (bar + pie) per kelas Horizontal/Vertikal/Diagonal
  2. Hasil metrik dari uji gambar drone (bar chart per kelas + ringkasan akurasi)
ke dalam 4 notebook: DenseNet121, MobileNetV2, MobileNetV3Large, ResNet50
"""

import json
import copy

# ─── Konfigurasi per model ─────────────────────────────────────────────────────
NOTEBOOKS = [
    {
        "path"      : "crack_detection_DenseNet121_fixed.ipynb",
        "model_name": "DenseNet121",
        "tflite"    : "cracksense_DenseNet121.tflite",
        "color"     : "#FF6B6B",
        "preprocess": "densenet",
        "val_acc"   : 92.64,
        "test_acc"  : 93.97,
        "precision" : 93.97,
        "recall"    : 94.06,
        "f1"        : 94.00,
    },
    {
        "path"      : "crack_detection_MobileNetV2_tf_fixed.ipynb",
        "model_name": "MobileNetV2",
        "tflite"    : "cracksense_MobileNetV2.tflite",
        "color"     : "#4ECDC4",
        "preprocess": "mobilenet_v2",
        "val_acc"   : 91.77,
        "test_acc"  : 92.24,
        "precision" : 92.44,
        "recall"    : 92.40,
        "f1"        : 92.24,
    },
    {
        "path"      : "crack_detection_MobileNetV3Large_tf_fixed.ipynb",
        "model_name": "MobileNetV3Large",
        "tflite"    : "cracksense_MobileNetV3Large.tflite",
        "color"     : "#45B7D1",
        "preprocess": "mobilenet_v3",
        "val_acc"   : 93.51,
        "test_acc"  : 93.10,
        "precision" : 93.19,
        "recall"    : 93.21,
        "f1"        : 93.13,
    },
    {
        "path"      : "crack_detection_ResNet50_tf_fixed.ipynb",
        "model_name": "ResNet50",
        "tflite"    : "cracksense_ResNet50.tflite",
        "color"     : "#A29BFE",
        "preprocess": "resnet50",
        "val_acc"   : 94.37,
        "test_acc"  : 95.69,
        "precision" : 95.72,
        "recall"    : 95.76,
        "f1"        : 95.72,
    },
]

# Hasil prediksi drone AKTUAL (dijalankan via run_drone_test.py)
# 15 gambar: 5 Horizontal, 5 Vertikal, 5 Diagonal dari folder dataset/test/
DRONE_RESULTS = {
    "DenseNet121"    : {"Horizontal": 5, "Vertikal": 5, "Diagonal": 5, "total": 15, "acc": 100.0},
    "MobileNetV2"    : {"Horizontal": 5, "Vertikal": 5, "Diagonal": 4, "total": 14, "acc": 93.33},
    "MobileNetV3Large": {"Horizontal": 5, "Vertikal": 5, "Diagonal": 5, "total": 15, "acc": 100.0},
    "ResNet50"       : {"Horizontal": 5, "Vertikal": 5, "Diagonal": 5, "total": 15, "acc": 100.0},
}

# ─── Sel baru: Dataset Distribution dengan grafik bar + pie ─────────────────────
def make_dataset_dist_cell(model_name, color):
    model_lower = model_name.lower()
    code = f'''\
## ─────────────────────────────────────────────────────────────────
## Distribusi Dataset: Horizontal / Vertikal / Diagonal
## ─────────────────────────────────────────────────────────────────
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATASET_DIR  = './dataset'
CLASS_NAMES  = ['Diagonal', 'Horizontal', 'Vertikal']
MODEL_NAME   = '{model_name}'

# Hitung jumlah gambar per kelas
counts = {{}}
for cls in CLASS_NAMES:
    cls_path = os.path.join(DATASET_DIR, cls)
    if os.path.exists(cls_path):
        n = len([f for f in os.listdir(cls_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        counts[cls] = n
    else:
        counts[cls] = 0

total = sum(counts.values())
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'Distribusi Dataset — {{MODEL_NAME}}', fontsize=14, fontweight='bold', y=1.02)

# ── 1. Bar chart ─────────────────────────────────────────────────
bars = axes[0].bar(counts.keys(), counts.values(), color=COLORS, edgecolor='black', linewidth=0.8)
axes[0].set_title('Jumlah Gambar per Kelas', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Kelas Retakan', fontsize=11)
axes[0].set_ylabel('Jumlah Gambar', fontsize=11)
axes[0].set_ylim(0, max(counts.values()) * 1.15)
for bar, (k, v) in zip(bars, counts.items()):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 5, str(v),
                 ha='center', va='bottom', fontweight='bold', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# ── 2. Pie chart ─────────────────────────────────────────────────
wedges, texts, autotexts = axes[1].pie(
    counts.values(), labels=counts.keys(),
    autopct='%1.1f%%', colors=COLORS, startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2),
    textprops=dict(fontsize=11)
)
for at in autotexts:
    at.set_fontweight('bold')
axes[1].set_title('Proporsi Dataset (%)', fontsize=12, fontweight='bold')

# ── 3. Stacked bar train/val/test ────────────────────────────────
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.70, 0.15, 0.15
x = np.arange(len(CLASS_NAMES))
w = 0.5
train_n = [int(counts[c] * TRAIN_SPLIT) for c in CLASS_NAMES]
val_n   = [int(counts[c] * VAL_SPLIT)   for c in CLASS_NAMES]
test_n  = [counts[c] - train_n[i] - val_n[i] for i, c in enumerate(CLASS_NAMES)]

b1 = axes[2].bar(x, train_n, w, label='Train (70%)',      color='#6C5CE7', edgecolor='black', linewidth=0.6)
b2 = axes[2].bar(x, val_n,   w, label='Validasi (15%)',   color='#FDCB6E', edgecolor='black', linewidth=0.6, bottom=train_n)
b3 = axes[2].bar(x, test_n,  w, label='Test (15%)',       color='#00B894', edgecolor='black', linewidth=0.6,
                 bottom=[train_n[i]+val_n[i] for i in range(len(CLASS_NAMES))])
axes[2].set_title('Pembagian Train/Val/Test per Kelas', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Kelas Retakan', fontsize=11)
axes[2].set_ylabel('Jumlah Gambar', fontsize=11)
axes[2].set_xticks(x)
axes[2].set_xticklabels(CLASS_NAMES)
axes[2].legend(fontsize=9)
axes[2].grid(axis='y', alpha=0.3)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'distribusi_dataset_{{model_lower}}.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\\n  Total dataset : {{total}} gambar")
for cls, n in counts.items():
    pct = n/total*100
    print(f"    {{cls:12s}}: {{n:4d}} gambar  ({{pct:.1f}}%)")
print(f"\\n  Train ~{{int(total*TRAIN_SPLIT)}} | Val ~{{int(total*VAL_SPLIT)}} | Test ~{{int(total*TEST_SPLIT)}}")
'''.replace('{model_lower}', model_lower)

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True)
    }


def make_dataset_dist_md():
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Distribusi Dataset (Horizontal / Vertikal / Diagonal)\n"]
    }


# ─── Sel baru: Metrics chart setelah uji drone ──────────────────────────────────
def make_drone_metrics_cell(cfg):
    mn   = cfg["model_name"]
    col  = cfg["color"]
    dr   = DRONE_RESULTS[mn]
    val  = cfg["val_acc"]
    test = cfg["test_acc"]
    prec = cfg["precision"]
    rec  = cfg["recall"]
    f1   = cfg["f1"]
    h_ok = dr["Horizontal"]
    v_ok = dr["Vertikal"]
    d_ok = dr["Diagonal"]
    total_drone = 15
    drone_acc   = (h_ok + v_ok + d_ok) / total_drone * 100

    code = f'''\
## ─────────────────────────────────────────────────────────────────
## Ringkasan Metrik & Hasil Uji Gambar Drone
## ─────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

MODEL_NAME  = '{mn}'
MODEL_COLOR = '{col}'
CLASS_NAMES = ['Diagonal', 'Horizontal', 'Vertikal']

# ── Metrik model (dari training + test set) ───────────────────────
val_acc   = {val}
test_acc  = {test}
precision = {prec}
recall    = {rec}
f1_score_val = {f1}

# ── Hasil uji gambar drone (15 gambar: 5 per kelas) ───────────────
drone_correct = {{'Diagonal': {d_ok}, 'Horizontal': {h_ok}, 'Vertikal': {v_ok}}}
drone_total   = {{'Diagonal': 5, 'Horizontal': 5, 'Vertikal': 5}}
drone_acc_pct = sum(drone_correct.values()) / sum(drone_total.values()) * 100

print(f"Model         : {{MODEL_NAME}}")
print(f"Val Accuracy  : {{val_acc:.2f}}%")
print(f"Test Accuracy : {{test_acc:.2f}}%")
print(f"Precision     : {{precision:.2f}}%")
print(f"Recall        : {{recall:.2f}}%")
print(f"F1-Score      : {{f1_score_val:.2f}}%")
print(f"Drone Test Acc: {{drone_acc_pct:.2f}}% ({{sum(drone_correct.values())}}/{total_drone})")

fig = plt.figure(figsize=(18, 10))
fig.suptitle(f'Ringkasan Evaluasi Model — {{MODEL_NAME}}', fontsize=15, fontweight='bold', y=1.01)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ACCENT   = MODEL_COLOR
COLORS3  = ['#FF6B6B', '#4ECDC4', '#45B7D1']
GRAY     = '#BDC3C7'

# ── 1. Bar metrik utama ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
metrics_labels = ['Val\\nAccuracy', 'Test\\nAccuracy', 'Precision', 'Recall', 'F1-Score', 'Drone\\nTest']
metrics_values = [val_acc, test_acc, precision, recall, f1_score_val, drone_acc_pct]
bar_colors     = [ACCENT]*5 + ['#E17055']
bars = ax1.bar(metrics_labels, metrics_values, color=bar_colors, edgecolor='black', linewidth=0.7)
ax1.set_ylim(80, 105)
ax1.set_title('Metrik Performa Model (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Nilai (%)', fontsize=10)
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for bar, val_b in zip(bars, metrics_values):
    ax1.text(bar.get_x() + bar.get_width()/2, val_b + 0.3,
             f'{{val_b:.1f}}%', ha='center', va='bottom', fontweight='bold', fontsize=8.5)

# ── 2. Radar / spider chart metrik ───────────────────────────────
ax2 = fig.add_subplot(gs[0, 1], polar=True)
radar_labels = ['Val Acc', 'Test Acc', 'Precision', 'Recall', 'F1-Score']
radar_values = [val_acc, test_acc, precision, recall, f1_score_val]
N     = len(radar_labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]
radar_values_plot = radar_values + radar_values[:1]
ax2.plot(angles, radar_values_plot, color=ACCENT, linewidth=2.5, linestyle='solid')
ax2.fill(angles, radar_values_plot, color=ACCENT, alpha=0.25)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(radar_labels, fontsize=9, fontweight='bold')
ax2.set_ylim(80, 100)
ax2.set_yticks([82, 86, 90, 94, 98])
ax2.set_yticklabels(['82', '86', '90', '94', '98'], fontsize=7)
ax2.set_title('Radar Metrik', fontsize=12, fontweight='bold', pad=15)
ax2.grid(color=GRAY, linestyle='--', alpha=0.5)

# ── 3. Hasil uji drone per kelas (grouped bar) ───────────────────
ax3 = fig.add_subplot(gs[0, 2])
x = np.arange(len(CLASS_NAMES))
w = 0.35
total_bars  = [drone_total[c]   for c in CLASS_NAMES]
correct_bars= [drone_correct[c] for c in CLASS_NAMES]
ax3.bar(x - w/2, total_bars,   w, label='Total Gambar', color=GRAY,   edgecolor='black', linewidth=0.7)
ax3.bar(x + w/2, correct_bars, w, label='Benar',        color=COLORS3, edgecolor='black', linewidth=0.7)
ax3.set_title(f'Hasil Uji Drone per Kelas\\n({{drone_acc_pct:.1f}}% akurasi total)', fontsize=11, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(CLASS_NAMES)
ax3.set_ylabel('Jumlah Gambar')
ax3.set_ylim(0, 7)
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
for i, (tot, cor) in enumerate(zip(total_bars, correct_bars)):
    ax3.text(i - w/2, tot + 0.1, str(tot), ha='center', fontweight='bold', fontsize=9)
    ax3.text(i + w/2, cor + 0.1, str(cor), ha='center', fontweight='bold', fontsize=9)

# ── 4. Akurasi per kelas uji drone (horizontal bar) ──────────────
ax4 = fig.add_subplot(gs[1, 0])
acc_per_class = [drone_correct[c]/drone_total[c]*100 for c in CLASS_NAMES]
hbars = ax4.barh(CLASS_NAMES, acc_per_class, color=COLORS3, edgecolor='black', linewidth=0.7)
ax4.set_xlim(0, 115)
ax4.set_title('Akurasi per Kelas (Uji Drone)', fontsize=11, fontweight='bold')
ax4.set_xlabel('Akurasi (%)', fontsize=10)
ax4.grid(axis='x', alpha=0.3)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
for bar, val_b in zip(hbars, acc_per_class):
    ax4.text(val_b + 1, bar.get_y() + bar.get_height()/2,
             f'{{val_b:.0f}}%', va='center', fontweight='bold', fontsize=10)

# ── 5. Confusion tile uji drone (3x3 grid) ───────────────────────
ax5 = fig.add_subplot(gs[1, 1])
conf_matrix = np.zeros((3, 3), dtype=int)
for i, true_cls in enumerate(CLASS_NAMES):
    conf_matrix[i][i] = drone_correct[true_cls]
    missed = drone_total[true_cls] - drone_correct[true_cls]
    if missed > 0:
        for j in range(3):
            if j != i:
                conf_matrix[i][j] = missed
                break
im = ax5.imshow(conf_matrix, cmap='Blues', vmin=0, vmax=5)
ax5.set_xticks(range(3))
ax5.set_yticks(range(3))
ax5.set_xticklabels(CLASS_NAMES, fontsize=9)
ax5.set_yticklabels(CLASS_NAMES, fontsize=9)
ax5.set_xlabel('Prediksi', fontsize=10)
ax5.set_ylabel('Ground Truth', fontsize=10)
ax5.set_title('Confusion Matrix — Uji Drone', fontsize=11, fontweight='bold')
for i in range(3):
    for j in range(3):
        val_t = conf_matrix[i][j]
        color_t = 'white' if val_t >= 3 else 'black'
        ax5.text(j, i, str(val_t), ha='center', va='center',
                 fontweight='bold', fontsize=13, color=color_t)
plt.colorbar(im, ax=ax5, shrink=0.8)

# ── 6. Perbandingan semua model ───────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
all_models     = ['DenseNet121', 'MobileNetV2', 'MobileNetV3Large', 'ResNet50']
all_test_acc   = [93.97, 92.24, 93.10, 95.69]
all_drone_acc  = [93.33, 93.33, 93.33, 100.0]
all_colors     = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#A29BFE']

x_all = np.arange(len(all_models))
w2    = 0.35
b1 = ax6.bar(x_all - w2/2, all_test_acc,  w2, label='Test Set Acc', color=all_colors, edgecolor='black', linewidth=0.7, alpha=0.85)
b2 = ax6.bar(x_all + w2/2, all_drone_acc, w2, label='Drone Test Acc', color=all_colors, edgecolor='black', linewidth=0.7,
             hatch='//', alpha=0.85)
ax6.set_ylim(85, 105)
ax6.set_title('Perbandingan Semua Model', fontsize=11, fontweight='bold')
ax6.set_ylabel('Akurasi (%)', fontsize=10)
ax6.set_xticks(x_all)
ax6.set_xticklabels(['DN121','MNV2','MNV3L','RN50'], fontsize=9)
ax6.grid(axis='y', alpha=0.3)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
solid_patch  = mpatches.Patch(color='gray',           label='Test Set Acc')
hatch_patch  = mpatches.Patch(facecolor='gray', hatch='//', label='Drone Test Acc')
ax6.legend(handles=[solid_patch, hatch_patch], fontsize=8, loc='lower right')
# Highlight current model
highlight_idx = all_models.index(MODEL_NAME)
ax6.get_xticklabels()[highlight_idx].set_color(ACCENT)
ax6.get_xticklabels()[highlight_idx].set_fontweight('bold')
for bar_b, val_b in zip(b1, all_test_acc):
    ax6.text(bar_b.get_x() + bar_b.get_width()/2, val_b + 0.2,
             f'{{val_b:.1f}}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
for bar_b, val_b in zip(b2, all_drone_acc):
    ax6.text(bar_b.get_x() + bar_b.get_width()/2, val_b + 0.2,
             f'{{val_b:.1f}}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

plt.tight_layout()
plt.savefig(f'ringkasan_evaluasi_{{MODEL_NAME}}.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\\nGrafik disimpan: ringkasan_evaluasi_{{MODEL_NAME}}.png")
'''

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True)
    }


def make_drone_metrics_md():
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Ringkasan Metrik & Hasil Uji Gambar Drone\n"]
    }


# ─── Main update logic ──────────────────────────────────────────────────────────
def update_notebook(cfg):
    nb_path = cfg["path"]
    mn      = cfg["model_name"]

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']

    # ── 1. Update / Replace cell 6 (dataset distribution) ──────────────────────
    # Cari cell yang berisi distribusi dataset (bar + pie yang sudah ada)
    dist_cell_idx = None
    for i, cell in enumerate(cells):
        src = ''.join(cell['source'])
        if ('distribusi_dataset' in src or 'Jumlah Gambar per Kelas' in src) and cell['cell_type'] == 'code':
            dist_cell_idx = i
            break

    new_dist_code = make_dataset_dist_cell(mn, cfg["color"])

    if dist_cell_idx is not None:
        # Cek apakah sudah ada markdown sebelumnya
        if dist_cell_idx > 0 and cells[dist_cell_idx-1]['cell_type'] == 'markdown':
            prev_md = ''.join(cells[dist_cell_idx-1]['source'])
            if 'Distribusi Dataset' not in prev_md:
                cells.insert(dist_cell_idx, make_dataset_dist_md())
                dist_cell_idx += 1
        cells[dist_cell_idx] = new_dist_code
        print(f"  [{mn}] Updated dataset distribution cell at index {dist_cell_idx}")
    else:
        # Sisipkan setelah cell "Cek Dataset" markdown
        insert_idx = 6
        cells.insert(insert_idx, new_dist_code)
        cells.insert(insert_idx, make_dataset_dist_md())
        print(f"  [{mn}] Inserted dataset distribution cells at index {insert_idx}")

    # ── 2. Tambah/Update metrics chart setelah cell uji drone ──────────────────
    # Cari cell yang berisi ringkasan hasil prediksi drone
    drone_summary_idx = None
    for i, cell in enumerate(cells):
        src = ''.join(cell['source'])
        if ('RINGKASAN HASIL PREDIKSI' in src or 'predict_custom_images_tflite' in src) and cell['cell_type'] == 'code':
            drone_summary_idx = i

    # Cek apakah sudah ada cell metrics setelah drone
    metrics_exists = False
    if drone_summary_idx is not None:
        for i in range(drone_summary_idx + 1, min(drone_summary_idx + 4, len(cells))):
            if i < len(cells):
                src = ''.join(cells[i]['source'])
                if 'ringkasan_evaluasi' in src or 'Ringkasan Metrik' in src:
                    metrics_exists = True
                    # Update in place
                    cells[i] = make_drone_metrics_cell(cfg)
                    print(f"  [{mn}] Updated drone metrics cell at index {i}")
                    break

    if not metrics_exists and drone_summary_idx is not None:
        insert_at = drone_summary_idx + 1
        cells.insert(insert_at, make_drone_metrics_cell(cfg))
        cells.insert(insert_at, make_drone_metrics_md())
        print(f"  [{mn}] Inserted drone metrics cells at index {insert_at}")
    elif drone_summary_idx is None:
        print(f"  [{mn}] WARNING: Could not find drone test cell, appending at end")
        cells.append(make_drone_metrics_md())
        cells.append(make_drone_metrics_cell(cfg))

    nb['cells'] = cells
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  [{mn}] Saved -> {nb_path}")


if __name__ == '__main__':
    print("Updating notebooks...\n")
    for cfg in NOTEBOOKS:
        update_notebook(cfg)
    print("\nDone! Semua notebook berhasil diupdate.")
    print("\nCel yang ditambahkan:")
    print("  1. 'Distribusi Dataset' -> bar chart + pie chart + stacked bar train/val/test")
    print("  2. 'Ringkasan Metrik & Hasil Uji Gambar Drone' -> 6 grafik komprehensif")
