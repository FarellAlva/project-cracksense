import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2

# 1. Configuration
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
DATASET_DIR = './dataset'
MODEL_NAME = 'crack_detection_mobilenetv2'

# GPU Memory Growth Setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 2. Data Augmentation
def get_augmentation_model():
    return tf.keras.Sequential([
        layers.RandomBrightness(0.2),
        layers.RandomContrast((0.8, 1.2)),
        layers.RandomRotation(0.05), # ±18 degrees approx, helps with angle boundaries
        layers.RandomZoom(0.1),      # Helps with varying crack widths
        layers.Resizing(IMG_SIZE, IMG_SIZE)
    ])

def preprocess_image(image, label):
    # Original notebook logic: crop then resize
    # Simplified here to keep quality; notebook used specific random crop
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# 3. Data Loading
print("Loading datasets...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.3, # Split 70/30 (Val will be split further if needed or used as is)
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.3,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Further split val_test_ds into Val and Test (15% each of total)
val_batches = tf.data.experimental.cardinality(val_test_ds)
test_ds = val_test_ds.take(val_batches // 2)
val_ds = val_test_ds.skip(val_batches // 2)

# Prefetch/Map
augmentation = get_augmentation_model()
train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y)).map(preprocess_image).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess_image).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess_image).prefetch(tf.data.AUTOTUNE)

# 4. Model Building
def build_production_model(num_classes=3, fine_tune_at=0):
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    if fine_tune_at == 0:
        base_model.trainable = False
        lr = 1e-4
    else:
        base_model.trainable = True
        # Fine-tune only the top layers
        for layer in base_model.layers[:-fine_tune_at]:
            layer.trainable = False
        lr = 1e-5 # Lower LR for fine-tuning
        
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model

# 5. Training Stage 1: Head Only
print("Starting Training Stage 1: Classifier Head...")
model = build_production_model(fine_tune_at=0)

train_callbacks = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=train_callbacks
)

# Training Stage 2: Fine-Tuning
print("\nStarting Training Stage 2: Fine-Tuning (Top 50 Layers)...")
# Re-build/Compile with fine-tuning enabled
model = build_production_model(fine_tune_at=50)
# Transfer weights from stage 1
model.load_weights(f'{MODEL_NAME}_best.keras') # Assuming checkpoint saved it

train_callbacks_ft = [
    callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    callbacks.ModelCheckpoint(f'{MODEL_NAME}_best.keras', save_best_only=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
]

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=train_callbacks_ft
)

# 6. Evaluation & Export
print("Evaluating on test set...")
loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc:.4f}")

# Save final model
model.save(f'{MODEL_NAME}_final.keras')

# Export to TFLite
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(f'{MODEL_NAME}.tflite', 'wb') as f:
    f.write(tflite_model)

print("Done!")
