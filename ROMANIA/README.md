Basically trains a neural network to recognize 3 types of drinks from photos:
- Cola (coca cola, pepsi, whatever)
- Fanta 
- Sprite

I need to get at least 90% accuracy or it doesn't count lol

## How to run

First install stuff:
```bash
pip install ultralytics torch torchvision
```

Put your images like this:
```
ROMANIA/
  dataset/
    cola/     <- put cola pics here
    fanta/    <- fanta pics
    sprite/   <- sprite pics
```

Then just run:
```bash
python training_model.py
```

It takes forever on CPU (like 2-3 hours) but way faster if you have a GPU.

## What happens when you run it

1. It splits your images into training and validation (80/20)
2. Downloads a pretrained YOLO model 
3. Trains for up to 300 epochs (but stops early if it's not improving)
4. Saves the best model as `model.pt`

## Settings I used

- YOLOv8 nano (the small/fast one)
- 224x224 images
- Batch size 16 (or 8 if cpu)
- A bunch of data augmentation stuff

## Problems I ran into

**PyTorch 2.6 issues**: Had to patch torch.load because of some weights_only thing. The script handles it automatically now.

**Low accuracy**: If you're not hitting 90%, you probably need more images. I had like 170 cola pics but only 9 fanta and 8 sprite which was NOT balanced at all. Try to get at least 100 of each.

**Out of memory**: Change `batch_size = 4` in the training script if your computer crashes.

## GPU stuff

For AMD:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
```

For NVIDIA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Files

- `training_model.py` - the main script that does everything
- `dataset/` - raw images go here
- `data/` - processed train/val splits (auto-generated)
- `results/` - training logs and graphs
- `model.pt` - the final trained model (this is what you submit)
