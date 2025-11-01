import cv2
import os 
import numpy as np
from scipy.spatial import distance
import sys

# Path to your video
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_PATH, "dataset")
video_path = os.path.join(DATASET_PATH, f"char_{sys.argv[1]}.mp4")
EVERY_N_FRAME = 24
# Video path is dataset/char_{first_arg}.mp4
def main():
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Read the first frame
    height, width = 400, 400
    black_img = np.zeros((height, width, 3), dtype=np.uint8)

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()  # read next frame
        if not ret:
            break  # end of video

        # Process the frame here
        # e.g., display or save it
        # cv2.imshow("Frame", frame)         
        if frame_count %EVERY_N_FRAME == 0:                                                                                                     
            midpoint = middle_point_from_image(frame)
            x, y = midpoint
            cv2.circle(black_img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # (B, G, R)

        frame_count += 1
        if frame_count % 1000 == 0:
            print(frame_count, "/", total_frames, f" %{frame_count/total_frames*100}")

    cap.release()
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(DATASET_PATH,f"letter{sys.argv[1]}.jpg"), black_img)


def middle_point_from_image(img):
    pixels = np.array(img)

    # Create a mask for non-black pixels
    mask = np.any(pixels != [0, 0, 0], axis=-1)

    # Extract only those pixels
    ys, xs = np.where(mask)
    coords = np.column_stack((xs, ys))

    dist_matrix = distance.cdist(coords, coords, 'euclidean')

    # Find the indices of the two farthest points
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)

    p1, p2 = coords[i], coords[j]

    # Compute the midpoint
    midpoint = (p1 + p2) / 2

    midpoint = np.array(midpoint, dtype=int)  # round or cast to int
    return midpoint
if __name__ == "__main__":
    main()