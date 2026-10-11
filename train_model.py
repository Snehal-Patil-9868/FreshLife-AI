import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = r"E:\FreshLife_AI"

TRAIN_DIR = os.path.join(BASE_DIR, "dataset_split", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset_split", "validation")
TEST_DIR = os.path.join(BASE_DIR, "dataset_split", "test")

MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "fruit_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training folder not found:\n{TRAIN_DIR}")

if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(f"Validation folder not found:\n{VAL_DIR}")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Test folder not found:\n{TEST_DIR}")

print("=" * 60)
print("FreshLife AI - Fruit Classification Model")
print("=" * 60)

# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses detected:")
for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")

print(f"\nTotal classes: {num_classes}")

# Save class names for Flask later
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=4)

print(f"\nClass names saved to:")
print(CLASS_NAMES_PATH)

# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
], name="data_augmentation")

# ============================================================
# MOBILE NET V2 BASE MODEL
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False

# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# MobileNetV2 expects pixel values in [-1, 1]
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel summary:")
model.summary()

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING MODEL ON TEST DATA")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(MODEL_PATH)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print(f"\nModel saved at:")
print(MODEL_PATH)

print(f"\nClass names saved at:")
print(CLASS_NAMES_PATH)

print("\nYour model classes are:")

for class_name in class_names:
    print(" -", class_name)

print("\nNext step:")
print("Connect this model to Flask and FreshLife AI UI.")