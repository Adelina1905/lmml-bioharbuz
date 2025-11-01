import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os


def load_model_and_data():
    """Load the model, class names, and input image"""
    model_path = './INDONESIA/my_model.h5'
    class_names_path = './INDONESIA/class_names.txt'
    image_path = './INDONESIA/cat_original_img.jpg'

    if not os.path.exists(model_path) or not os.path.exists(class_names_path):
        print(f"❌ Error: Model or class names not found.")
        print(f"💡 Please run train_classifier.py first to create '{model_path}' and '{class_names_path}'.")
        exit()

    if not os.path.exists(image_path):
        print(f"❌ Error: Input image not found at '{image_path}'.")
        exit()

    model = keras.models.load_model(model_path)
    
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    img = Image.open(image_path)
    img = img.resize((224, 224))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img).astype('float32') / 255.0
    
    return model, class_names, img_array


def create_optimal_perturbation(model, img_array, target_class_idx, 
                               epsilon=0.2, iterations=500, alpha=0.01):
    """
    Create optimal perturbation using PGD with momentum
    This perturbation should be transferable to other models
    
    Args:
        model: The classifier model
        img_array: Original image (224, 224, 3) normalized to [0, 1]
        target_class_idx: Index of target class (panda)
        epsilon: Maximum perturbation magnitude
        iterations: Number of iterations
        alpha: Step size
    
    Returns:
        perturbation: The adversarial perturbation
    """
    # Initialize with larger random noise to start closer to bounds
    perturbation = np.random.uniform(-epsilon/2, epsilon/2, size=img_array.shape).astype('float32')
    
    # Momentum buffer for smoother updates
    momentum = np.zeros_like(perturbation)
    decay = 0.9
    
    print(f"🎯 Creating optimal perturbation...")
    print(f"   Epsilon: {epsilon}")
    print(f"   Iterations: {iterations}")
    print(f"   Alpha: {alpha}\n")
    
    best_confidence = 0.0
    best_perturbation = perturbation.copy()
    
    for i in range(iterations):
        # Create adversarial example
        adversarial = np.clip(img_array + perturbation, 0, 1)
        adversarial_tensor = tf.constant(adversarial, dtype=tf.float32)
        adversarial_tensor = tf.expand_dims(adversarial_tensor, 0)
        
        # Compute gradient
        with tf.GradientTape() as tape:
            tape.watch(adversarial_tensor)
            predictions = model(adversarial_tensor, training=False)
            # Maximize target class probability
            target_loss = predictions[0][target_class_idx]
        
        gradient = tape.gradient(target_loss, adversarial_tensor)
        grad_np = gradient.numpy()[0]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_np)
        if grad_norm > 0:
            grad_np = grad_np / grad_norm
        
        # Update with momentum - more aggressive
        momentum = decay * momentum + grad_np
        perturbation += alpha * np.sign(momentum)
        
        # Project to epsilon ball - this ensures full [-epsilon, epsilon] range
        perturbation = np.clip(perturbation, -epsilon, epsilon)
        
        # Check progress every 25 iterations
        if (i + 1) % 25 == 0 or i == 0:
            # For testing, clip the result
            test_adversarial = np.clip(img_array + perturbation, 0, 1)
            predictions = model.predict(test_adversarial[np.newaxis, ...], verbose=0)
            confidence = predictions[0][target_class_idx]
            
            # Track best
            if confidence > best_confidence:
                best_confidence = confidence
                best_perturbation = perturbation.copy()
            
            print(f"Iter {i+1:4d}/{iterations}: Current = {confidence:.4f} ({confidence*100:.1f}%) | Best = {best_confidence:.4f} ({best_confidence*100:.1f}%)")
            print(f"   Perturbation range: [{perturbation.min():.4f}, {perturbation.max():.4f}]")
            
            # Early stopping if we achieve very high confidence
            if best_confidence >= 0.95:
                print(f"\n🎉 Achieved excellent confidence {best_confidence:.4f}!")
                return best_perturbation
    
    return best_perturbation


def main():
    print("=" * 60)
    print("🐼 OPTIMAL PANDA PERTURBATION GENERATOR 🐼")
    print("=" * 60)
    
    # Load model and data
    print("\n📂 Loading model and data...")
    model, class_names, img_array = load_model_and_data()
    
    if 'panda' not in class_names:
        print("❌ Error: 'panda' class not found in the dataset.")
        print("💡 Make sure the training data contains a 'panda' directory.")
        return

    panda_idx = class_names.index('panda')
    print(f"Target class: 'panda' (index {panda_idx})")
    
    # Original prediction
    print("\n🔍 Original prediction:")
    original_pred = model.predict(img_array[np.newaxis, ...], verbose=0)
    original_class = np.argmax(original_pred[0])
    print(f"   Class: '{class_names[original_class]}' ({original_pred[0][original_class]:.4f})")
    print(f"   Panda: {original_pred[0][panda_idx]:.4f}\n")
    
    # Create optimal perturbation
    perturbation = create_optimal_perturbation(
        model=model,
        img_array=img_array,
        target_class_idx=panda_idx,
        epsilon=0.2,
        iterations=500,
        alpha=0.01
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
    adversarial = np.clip(img_array + perturbation, 0, 1)
    adversarial_pred = model.predict(adversarial[np.newaxis, ...], verbose=0)
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
        print(f"✅ Good! Reached {panda_confidence:.4f} confidence")
    else:
        print(f"⚠️  Reached {panda_confidence:.4f} confidence")
        print("💡 Try training model longer or adjusting parameters")
    print("=" * 60)
    
    # Save to root directory as required by the task
    np.save('perturbation.npy', perturbation)
    print(f"\n💾 Saved: perturbation.npy (in root directory)")
    
    # Also save to INDONESIA folder for local testing
    np.save('./INDONESIA/perturbation.npy', perturbation)
    
    # Save visualization images
    adversarial_img = (adversarial * 255).astype(np.uint8)
    Image.fromarray(adversarial_img).save('./INDONESIA/adversarial_image.jpg')
    
    perturbation_vis = ((perturbation - perturbation.min()) / 
                       (perturbation.max() - perturbation.min() + 1e-8) * 255).astype(np.uint8)
    Image.fromarray(perturbation_vis).save('./INDONESIA/perturbation_visualization.jpg')
    
    print(f"🖼️  Saved: ./INDONESIA/adversarial_image.jpg")
    print(f"🎨 Saved: ./INDONESIA/perturbation_visualization.jpg")


if __name__ == "__main__":
    main()