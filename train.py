from pathlib import Path
import json
import tensorflow as tf

# ---------- SETTINGS ----------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
DATABASE_DIR = BASE_DIR / "database"

MODEL_PATH = DATABASE_DIR / "enervision_appliances.keras"
CLASS_NAMES_PATH = DATABASE_DIR / "class_names.json"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 10
SEED = 42

CLASS_NAMES = [
    "AC",
    "Air_Cooler",
    "Fan",
    "Light_Bulb",
    "Microwave",
    "TV",
    "Washing_Machine",
]

tf.keras.utils.set_random_seed(SEED)


def main():
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    # Verify that every required folder exists.
    for name in CLASS_NAMES:
        folder = DATASET_DIR / name
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing dataset folder: {folder}"
            )

        images = [
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }
        ]

        if len(images) < 2:
            raise ValueError(
                f"{name} needs at least 2 images; found {len(images)}."
            )

        print(f"{name}: {len(images)} images")

    # Load images and split into training and validation sets.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        label_mode="int",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        label_mode="int",
    )

    print("Class order:", train_ds.class_names)

    # MobileNetV2 preprocessing is INCLUDED in this model.
    # Do not apply preprocess_input again in test_appliance.py.
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)

    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1
    )(x)

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES),
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, indent=4)

    print("\nTraining completed!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Class names saved to: {CLASS_NAMES_PATH}")


if __name__ == "__main__":
    main()