# Ukraine Solution - Track the Thief
you can run the solution.py with a number argument to select the video to process.
# Example:
```bash
python solution.py 1
```
selects the dataset/char_1.mp4 video file to process.

### How the solution works:
1. The video is read frame by frame.
2. The pixels that are not black are extracted.
3. We found their center point.
4. We draw a red circle at the center point on a black image.
5. Finally, we save the resulting image as letter_n.jpg in the dataset folder.