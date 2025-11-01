import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

def create_classifier(num_classes):
    """Creates a simple CNN classifier model."""
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Rescaling(1./255),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    print("=" * 60)
    print("🐾 TRAINING ANIMAL CLASSIFIER 🐾")
    print("=" * 60)

    # --- Configuration ---
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 5  # As requested for local testing
    DATA_DIR = './INDONESIA/dataset/animals/animals'
    MODEL_SAVE_PATH = './INDONESIA/my_model.h5'
    CLASS_NAMES_PATH = './INDONESIA/class_names.txt'

    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Dataset directory not found at {DATA_DIR}")
        print("💡 Please run dataset_import.py first to download the data.")
        return

    # --- Load Data ---
    print(f"\n📂 Loading data from {DATA_DIR}...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)
    print(f"   Found {num_classes} classes: {class_names}")

    # --- Create and Compile Model ---
    print("\n🏗️  Creating and compiling model...")
    model = create_classifier(num_classes)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # --- Train Model ---
    print(f"\n🎓 Training for {EPOCHS} epochs...")
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS
    )

    # --- Save Model and Class Names ---
    print("\n💾 Saving model and class names...")
    model.save(MODEL_SAVE_PATH)
    print(f"   Model saved to {MODEL_SAVE_PATH}")

    with open(CLASS_NAMES_PATH, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    print(f"   Class names saved to {CLASS_NAMES_PATH}")

    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()