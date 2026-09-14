import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuration
IMG_SIZE = 224
BATCH_SIZE = 16
DATASET_DIR = './dataset'
MODEL_PATH = 'crack_detection_mobilenetv2_best.keras'

# 2. Data Loading (Test Set)
print("Loading test data...")
# We use the same seed as train.py to ensure the same split
val_test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.3,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_batches = tf.data.experimental.cardinality(val_test_ds)
test_ds = val_test_ds.take(val_batches // 2)

def preprocess_image(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

test_ds = test_ds.map(preprocess_image).prefetch(tf.data.AUTOTUNE)

# 3. Load Model
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# 4. Predictions
print("Generating predictions...")
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 5. Metrics
class_names = val_test_ds.class_names
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# 6. Confusion Matrix Plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Crack Detection')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved to confusion_matrix.png")
