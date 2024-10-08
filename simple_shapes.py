import numpy as np
import cv2
import random
import datetime
import h5py

def create_frame_with_blobs(frame_size, blob_x_size_range, blob_y_size_range, num_blobs_range):
    # Unpack frame size
    height, width = frame_size

    # Create a black frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Sample the number of blobs
    num_blobs = random.randint(*num_blobs_range)

    for _ in range(num_blobs):
        # Sample blob size
        blob_x_size = random.uniform(*blob_x_size_range)
        blob_y_size = random.uniform(*blob_y_size_range)

        # Sample blob center
        center_x = random.randint(0, width - 1)
        center_y = random.randint(0, height - 1)

        # Draw the ellipse
        cv2.ellipse(
            frame,
            (center_x, center_y),
            (int(blob_x_size / 2), int(blob_y_size / 2)),
            angle=random.uniform(0, 360),
            startAngle=0,
            endAngle=360,
            color=(255, 255, 255),
            thickness=-1
        )

    return frame[:,:,0]

# Parameters
base_save_dir = '/home/aabdalq/projects/personal/fluoseg/results/generated/control'
n_frames=1000
frame_size = (200, 320)          # Size of the frame as [height, width]
blob_x_size_range = (10, 20)     # Range for blob x size as [min, max]
blob_y_size_range = (10, 25)     # Range for blob y size as [min, max]
num_blobs_range = (8, 15)        # Range for the number of blobs as [min, max]

# Create frame with blobs


# get current time in yyyy-mm-dd-hh-mm-ss format
current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
filepath = f'{base_save_dir}/{current_time}.h5'


all_frames = np.zeros((n_frames, *frame_size), dtype=np.uint8)
for i in range(n_frames):
    frame = create_frame_with_blobs(
        frame_size=frame_size,
        blob_x_size_range=blob_x_size_range,
        blob_y_size_range=blob_y_size_range,
        num_blobs_range=num_blobs_range
    )
    all_frames[i] = frame

with h5py.File(filepath, 'w') as f:
    f.create_dataset('mov', data=all_frames)