from ultralytics import YOLO
import os
from pathlib import Path
import shutil
import torch
import torch.nn as nn
import urllib.request

# Import the Ultralytics modules to patch
try:
    from ultralytics.nn import tasks 
    from ultralytics.utils import torch_utils
except ImportError:
    print("❌ Error: Could not import ultralytics modules. Ensure ultralytics is installed.")

print("✓ Preparing environment for PyTorch 2.6+ compatibility.")

# --- COMPREHENSIVE PATCH for PyTorch 2.6 and Ultralytics ---
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=None, **kwargs):
    """Patched torch.load that always sets weights_only=False for compatibility."""
    # Force weights_only=False for all ultralytics operations
    return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **kwargs)

# Replace torch.load globally
torch.load = patched_torch_load
print("✓ Global torch.load patched with weights_only=False.")

# Also patch the specific ultralytics functions
def patched_torch_safe_load(file, map_location=None, **kwargs):
    """Safely loads a PyTorch checkpoint with weights_only=False."""
    if map_location is None:
        map_location = 'cpu'
    kwargs['weights_only'] = False
    return torch.load(file, map_location=map_location, **kwargs), file

try:
    tasks.torch_safe_load = patched_torch_safe_load
    print("✓ Ultralytics torch_safe_load patched.")
except (NameError, AttributeError):
    pass
# ----------------------------------------------------

def check_gpu_availability():
    """Check and display GPU availability"""
    print("\n🖥️  GPU Check:")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"   ✅ CUDA GPU detected: {gpu_count} device(s)")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"      GPU {i}: {gpu_name}")
        device = 'cuda:0'
    else:
        print("   ⚠️  No CUDA GPU detected")
        print("   💡 For AMD GPUs, you need PyTorch with ROCm support")
        print("   💡 Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2")
        print("   ℹ️  Training will use CPU (slower)")
        device = 'cpu'
    
    print(f"   Device selected: {device}")
    return device

def download_fresh_model():
    """Download a fresh yolov8n-cls.pt model compatible with current ultralytics"""
    cache_dir = Path.home() / '.cache' / 'ultralytics'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = cache_dir / 'yolov8n-cls.pt'
    
    # Remove old incompatible model
    if model_path.exists():
        print("   🗑️  Removing old cached model...")
        model_path.unlink()
    
    # Download fresh model from official source
    print("   📥 Downloading fresh YOLOv8n-cls model...")
    url = 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-cls.pt'
    
    try:
        urllib.request.urlretrieve(url, model_path)
        print("   ✓ Model downloaded successfully")
        return str(model_path)
    except Exception as e:
        print(f"   ⚠️  Download failed: {e}")
        return None

def prepare_dataset():
    """Prepare dataset by organizing into train/val structure"""
    base_dir = Path('./ROMANIA')
    dataset_dir = base_dir / 'dataset'
    data_dir = base_dir / 'data'
    
    classes = ['cola', 'fanta', 'sprite']
    
    print("\n📁 Preparing dataset structure...")
    
    # Create train/val directories
    for split in ['train', 'val']:
        for cls in classes:
            (data_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Check source directories and split data
    total_images = 0
    for cls in classes:
        src_dir = dataset_dir / cls
        
        if not src_dir.exists():
            print(f"   ⚠️  Warning: {src_dir} not found")
            continue
        
        # Get all images
        images = list(src_dir.glob('*.jpg')) + list(src_dir.glob('*.png')) + list(src_dir.glob('*.jpeg'))
        
        if not images:
            print(f"   ⚠️  Warning: No images found in {src_dir}")
            continue
        
        # Split 80/20 train/val
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy to train directory
        for img in train_images:
            dst = data_dir / 'train' / cls / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
        
        # Copy to val directory
        for img in val_images:
            dst = data_dir / 'val' / cls / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
        
        total_images += len(images)
        print(f"   ✓ {cls}: {len(train_images)} train, {len(val_images)} val (total: {len(images)})")
    
    return data_dir, total_images

def train_yolo_classifier():
    """Train YOLOv8 classification model for soft drink brands"""
    
    print("=" * 60)
    print("🥤 YOLO SOFT DRINK CLASSIFIER TRAINING 🥤")
    print("=" * 60)
    
    # Check GPU availability
    device = check_gpu_availability()
    
    # Prepare dataset
    data_dir, total_images = prepare_dataset()
    
    if total_images == 0:
        print("\n❌ Error: No training images found! Ensure they are in ./ROMANIA/dataset/{cola, fanta, sprite}/")
        return
    
    print(f"\n✅ Total images prepared: {total_images}")
    
    # Count images in train/val splits
    classes = ['cola', 'fanta', 'sprite']
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    
    total_train = sum(len(list((train_dir / cls).glob('*'))) for cls in classes)
    total_val = sum(len(list((val_dir / cls).glob('*'))) for cls in classes)

    print("\n📊 Dataset statistics:")
    print(f"   Training: {total_train} images")
    print(f"   Validation: {total_val} images")
    
    # Show per-class distribution
    print("\n   Per-class distribution:")
    for cls in classes:
        train_count = len(list((train_dir / cls).glob('*')))
        val_count = len(list((val_dir / cls).glob('*')))
        print(f"      {cls}: {train_count} train, {val_count} val")
    
    # Initialize YOLOv8 classification model
    print("\n🤖 Loading YOLOv8 classification model...")
    
    # Download fresh compatible model
    model_path = download_fresh_model()
    
    if model_path is None:
        print("   ⚠️  Using 'yolov8n-cls' (will auto-download)")
        model_path = 'yolov8n-cls'
    
    try:
        model = YOLO(model_path)
        print("   ✓ Model loaded (YOLOv8n-cls)")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        print("   💡 Trying alternative: training from scratch...")
        # Train from scratch as fallback
        model = YOLO('yolov8n-cls.yaml')
        print("   ✓ Model initialized from config (training from scratch)")
    
    # Adjust batch size based on device
    if device == 'cpu':
        batch_size = 8  # Smaller batch for CPU
        print("   💡 Using smaller batch size (8) for CPU training")
    else:
        batch_size = 16  # Standard batch for GPU
    
    # Training configuration - INCREASED FOR BETTER PERFORMANCE
    IMG_SIZE = 224
    EPOCHS = 100  
    PATIENCE = 50  # ⬆️ INCREASED early stopping patience
    
    print("\n⚙️  Training configuration:")
    print(f"   Device: {device}")
    print(f"   Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {EPOCHS} (increased for better convergence)")
    print(f"   Early stopping patience: {PATIENCE}")
    print(f"   Optimizer: SGD with cosine LR scheduler")
    print(f"   Data augmentation: Enabled")
    
    # Train the model
    print("\n🎓 Starting training...")
    print("=" * 60)
    
    try:
        results = model.train(
            data=str(data_dir),
            epochs=EPOCHS,  # Increased epochs
            imgsz=IMG_SIZE,
            batch=batch_size,
            device=device,
            project=str(Path('./ROMANIA/results')),
            name='softdrink_classifier',
            exist_ok=True,
            patience=PATIENCE,  # Increased patience
            save=True,
            plots=True,
            verbose=True,
            workers=4,
            # Additional parameters for better training
            lr0=0.01,  # Initial learning rate
            lrf=0.01,  # Final learning rate (lr0 * lrf)
            momentum=0.937,  # SGD momentum
            weight_decay=0.0005,  # Weight decay
            warmup_epochs=3.0,  # Warmup epochs
            warmup_momentum=0.8,  # Warmup momentum
            cos_lr=True,  # Use cosine learning rate scheduler
            # Data augmentation
            hsv_h=0.015,  # HSV-Hue augmentation
            hsv_s=0.7,  # HSV-Saturation augmentation
            hsv_v=0.4,  # HSV-Value augmentation
            degrees=0.0,  # Rotation augmentation
            translate=0.1,  # Translation augmentation
            scale=0.5,  # Scale augmentation
            flipud=0.0,  # Vertical flip probability
            fliplr=0.5,  # Horizontal flip probability
        )
        
        print("\n" + "=" * 60)
        print("✅ Training completed!")
        
        # Save the best model as model.pt
        base_dir = Path('./ROMANIA')
        best_model_path = base_dir / 'results' / 'softdrink_classifier' / 'weights' / 'best.pt'
        final_model_path = base_dir / 'model.pt'
        
        if best_model_path.exists():
            shutil.copy(best_model_path, final_model_path)
            print(f"💾 Model saved to: {final_model_path}")
        else:
            print("⚠️  Warning: Best model not found at expected location.")
            print(f"   Checking for last.pt...")
            last_model_path = base_dir / 'results' / 'softdrink_classifier' / 'weights' / 'last.pt'
            if last_model_path.exists():
                shutil.copy(last_model_path, final_model_path)
                print(f"💾 Model saved from last.pt to: {final_model_path}")
        
        # Validation and Accuracy Check
        if total_val > 0:
            print("\n📈 Running final validation...")
            try:
                metrics = model.val()
                
                print("\n📊 Final Metrics:")
                if hasattr(metrics, 'top1'):
                    accuracy = metrics.top1
                    print(f"   Top-1 Accuracy: {accuracy:.2f}%")
                    
                    if accuracy >= 90:
                        print("\n🎉 SUCCESS! Accuracy ≥ 90% achieved!")
                        print("   🏆 Model ready for submission!")
                    elif accuracy >= 85:
                        print("\n✅ Very close! Consider:")
                        print("      - Collecting a few more diverse images")
                        print("      - Running training again (random seed variation)")
                    else:
                        print("\n⚠️  Model accuracy below target.")
                        print("   💡 Current accuracy: {:.2f}%".format(accuracy))
                        print("   💡 Target: 90%")
            except Exception as e:
                print(f"   ⚠️  Validation failed: {e}")
                print("   But model training completed successfully!")
        
        print("\n" + "=" * 60)
        print("🏁 Training pipeline completed!")
        print(f"📦 Submit file: {final_model_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        # Try to save model anyway if it exists
        base_dir = Path('./ROMANIA')
        best_model_path = base_dir / 'results' / 'softdrink_classifier' / 'weights' / 'best.pt'
        final_model_path = base_dir / 'model.pt'
        if best_model_path.exists():
            shutil.copy(best_model_path, final_model_path)
            print(f"\n💾 Despite error, model was saved to: {final_model_path}")
            print("   You can use this model for submission!")
        raise

def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("🚀 YOLO Soft Drink Classifier - Training Script")
    print("=" * 60)
    
    # Display PyTorch info
    print(f"\n📦 Environment:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
    
    try:
        train_yolo_classifier()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("\n💡 If training completed but saving failed, check:")
        print("   ./ROMANIA/results/softdrink_classifier/weights/best.pt")
        print("   You can manually copy it to model.pt")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()