import cv2
import os
import numpy as np

# Base path where cropped folders are stored
base_path = "special-rtdetr"

# Loop through each model folder inside cropped_pieces
for model_name in os.listdir(base_path):
    model_dir = os.path.join(base_path, model_name)
    if not os.path.isdir(model_dir):
        continue

    # Prepare a list to hold all images
    all_images = []

    # Sort and read all images
    for filename in sorted(os.listdir(model_dir)):
        filepath = os.path.join(model_dir, filename)
        img = cv2.imread(filepath)
        if img is not None:
            all_images.append(img)

    # Resize all images in a group to the same width and height
    def resize_images_uniformly(images):
        if not images:
            return []
        min_width = min(img.shape[1] for img in images)
        min_height = min(img.shape[0] for img in images)
        resized = [cv2.resize(img, (min_width, min_height)) for img in images]
        return resized

    # Merge all images vertically
    if all_images:
        all_images_resized = resize_images_uniformly(all_images)
        merged_images = np.vstack(all_images_resized)
        output_path = os.path.join(model_dir, "merged_images.jpg")
        cv2.imwrite(output_path, merged_images)
        print(f"Saved merged image to {output_path}")
