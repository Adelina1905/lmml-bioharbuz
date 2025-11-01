import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import os


def load_model_and_data():
    """Load the trained classifier and target image"""
    # Try different possible paths
    model_paths = [
        './INDONESIA/my_model.h5',
        './INDONESIA/pd_model.h5',
        'my_model.h5',
        'pd_model.h5'
    ]
    
    model = None
    for path in model_paths:
        if os.path.exists(path):
            print(f"   Found model at: {path}")
            model = keras.models.load_model(path)
            break
    
    if model is None:
        raise FileNotFoundError("No model file found. Please ensure my_model.h5 or pd_model.h5 exists.")
    
    with open('./INDONESIA/class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Load cat image
    img = Image.open('./INDONESIA/cat_original_img.jpg')
    img = img.resize((224, 224))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img).astype('float32') / 255.0
    
    return model, class_names, img_array


def load_panda_examples(num_samples=10):
    """Load real panda images to learn from"""
    panda_dirs = [
        './INDONESIA/dataset/animals/animals/panda',
        './INDONESIA/dataset/panda',
        './dataset/animals/animals/panda',
        './dataset/panda'
    ]
    
    panda_images = []
    
    for data_dir in panda_dirs:
        if os.path.exists(data_dir):
            print(f"   Found panda directory: {data_dir}")
            files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:num_samples]
            
            for file in files:
                try:
                    img = Image.open(os.path.join(data_dir, file))
                    img = img.resize((224, 224))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_array = np.array(img).astype('float32') / 255.0
                    panda_images.append(img_array)
                except Exception as e:
                    print(f"   ⚠️  Error loading {file}: {e}")
            
            if panda_images:
                return np.array(panda_images)
    
    print(f"⚠️  No panda images found in any directory")
    return None


class PerturbationGenerator(keras.Model):
    """Neural network that generates perturbations"""
    def __init__(self, epsilon=0.2):
        super().__init__()
        self.epsilon = epsilon
        
        # Simple encoder-decoder
        self.conv1 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv4 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv5 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv6 = layers.Conv2D(3, 3, padding='same', activation='tanh')
    
    def call(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        perturbation = self.conv6(h)
        
        # Scale to epsilon range
        perturbation = perturbation * self.epsilon
        
        return perturbation


def train_perturbation_generator(classifier, cat_image, panda_images, 
                                 target_class_idx, epsilon=0.2, epochs=1000):
    """Train a neural network to generate optimal perturbation"""
    
    print("=" * 60)
    print("🎓 TRAINING PERTURBATION GENERATOR")
    print("=" * 60)
    
    # Create generator
    generator = PerturbationGenerator(epsilon=epsilon)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    # Prepare cat image
    cat_batch = tf.constant(cat_image[np.newaxis, ...], dtype=tf.float32)
    
    best_confidence = 0.0
    best_perturbation = None
    no_improvement = 0
    
    print(f"\n🎯 Training for up to {epochs} epochs...")
    print(f"   Target: Panda class (index {target_class_idx})")
    print(f"   Epsilon: {epsilon}\n")
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            # Generate perturbation
            perturbation = generator(cat_batch, training=True)
            
            # Create adversarial image
            adversarial = tf.clip_by_value(cat_batch + perturbation, 0, 1)
            
            # Get classifier predictions
            predictions = classifier(adversarial, training=False)
            
            # Loss 1: Maximize target class probability (main objective)
            target_loss = -tf.reduce_mean(predictions[:, target_class_idx])
            
            # Loss 2: Minimize perturbation magnitude (keep it small)
            l2_loss = 0.005 * tf.reduce_mean(tf.square(perturbation))
            
            # Loss 3: Style loss with real pandas (if available)
            style_loss = 0.0
            if panda_images is not None and len(panda_images) > 0:
                # Use a few random panda samples
                num_samples = min(3, len(panda_images))
                panda_batch = tf.constant(panda_images[:num_samples], dtype=tf.float32)
                panda_features = classifier(panda_batch, training=False)
                
                # Match feature distributions
                style_loss = 0.05 * tf.reduce_mean(tf.square(
                    tf.reduce_mean(predictions, axis=0) - 
                    tf.reduce_mean(panda_features, axis=0)
                ))
            
            # Total loss
            total_loss = target_loss + l2_loss + style_loss
        
        # Update generator
        gradients = tape.gradient(total_loss, generator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
        
        # Track progress
        if (epoch + 1) % 50 == 0 or epoch == 0:
            confidence = predictions[0][target_class_idx].numpy()
            l2_norm = tf.norm(perturbation).numpy()
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_perturbation = perturbation[0].numpy()
                no_improvement = 0
            else:
                no_improvement += 1
            
            print(f"Epoch {epoch+1:4d}/{epochs}: "
                  f"Panda = {confidence:.4f} ({confidence*100:.1f}%) | "
                  f"Best = {best_confidence:.4f} ({best_confidence*100:.1f}%) | "
                  f"L2 = {l2_norm:.4f}")
            
            # Early stopping conditions
            if confidence >= 0.80:
                print(f"\n✅ SUCCESS! Reached target confidence {confidence:.4f}!")
                return perturbation[0].numpy()
            
            if no_improvement >= 10:
                print(f"\n⏹️  No improvement for 10 checks, stopping...")
                break
    
    if best_perturbation is not None:
        return best_perturbation
    else:
        return perturbation[0].numpy()


def main():
    print("=" * 60)
    print("🐼 LEARNED PERTURBATION ATTACK 🐼")
    print("=" * 60)
    
    # Load classifier and data
    print("\n📂 Loading classifier and data...")
    try:
        classifier, class_names, cat_image = load_model_and_data()
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\n💡 Please ensure one of these files exists:")
        print("   - ./INDONESIA/my_model.h5")
        print("   - ./INDONESIA/pd_model.h5")
        return
    
    panda_idx = class_names.index('panda')
    print(f"   Target class: 'panda' (index {panda_idx})")
    
    # Load panda examples
    print("\n🐼 Loading panda examples...")
    panda_images = load_panda_examples()
    if panda_images is not None:
        print(f"   ✅ Loaded {len(panda_images)} panda images for style learning")
    else:
        print("   ⚠️  No panda images found, using target-only approach")
    
    # Original prediction
    print("\n🔍 Original prediction:")
    original_pred = classifier.predict(cat_image[np.newaxis, ...], verbose=0)
    original_class = np.argmax(original_pred[0])
    print(f"   Class: '{class_names[original_class]}' ({original_pred[0][original_class]:.4f})")
    print(f"   Panda: {original_pred[0][panda_idx]:.4f}")
    
    # Train perturbation generator
    perturbation = train_perturbation_generator(
        classifier=classifier,
        cat_image=cat_image,
        panda_images=panda_images,
        target_class_idx=panda_idx,
        epsilon=0.2,
        epochs=1000
    )
    
    # Statistics
    l2_norm = np.linalg.norm(perturbation)
    print(f"\n📊 Perturbation statistics:")
    print(f"   L2 norm: {l2_norm:.4f}")
    print(f"   Range: [{perturbation.min():.4f}, {perturbation.max():.4f}]")
    print(f"   Mean: {perturbation.mean():.4f}")
    print(f"   Std: {perturbation.std():.4f}")
    
    # Test adversarial image
    print("\n🎭 Testing adversarial image...")
    adversarial = np.clip(cat_image + perturbation, 0, 1)
    adversarial_pred = classifier.predict(adversarial[np.newaxis, ...], verbose=0)
    adversarial_class = np.argmax(adversarial_pred[0])
    panda_confidence = adversarial_pred[0][panda_idx]
    
    print(f"   Predicted: '{class_names[adversarial_class]}'")
    print(f"   Panda confidence: {panda_confidence:.4f} ({panda_confidence*100:.1f}%)")
    
    # Result
    print("\n" + "=" * 60)
    if panda_confidence >= 0.80:
        print("🎉 SUCCESS! Achieved ≥ 80% confidence!")
        print(f"🚩 FLAG: SIGMOID_ADVERSARIAL")
    elif panda_confidence >= 0.50:
        print(f"✅ Good progress! Reached {panda_confidence:.4f} confidence")
        print("💡 Try increasing epsilon or epochs for better results")
    else:
        print(f"⚠️  Reached {panda_confidence:.4f} confidence")
        print("💡 Try adjusting epsilon, epochs, or learning rate")
    print("=" * 60)
    
    # Save
    np.save('./INDONESIA/perturbation.npy', perturbation)
    print(f"\n💾 Saved: ./INDONESIA/perturbation.npy")
    
    # Save images
    adversarial_img = (adversarial * 255).astype(np.uint8)
    Image.fromarray(adversarial_img).save('./INDONESIA/adversarial_image.jpg')
    
    perturbation_vis = ((perturbation - perturbation.min()) / 
                       (perturbation.max() - perturbation.min() + 1e-8) * 255).astype(np.uint8)
    Image.fromarray(perturbation_vis).save('./INDONESIA/perturbation_visualization.jpg')
    
    print(f"🖼️  Saved: adversarial_image.jpg")
    print(f"🎨 Saved: perturbation_visualization.jpg")


if __name__ == "__main__":
    main()