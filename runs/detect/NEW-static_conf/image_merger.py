import cv2
import os
import numpy as np

# Base path where cropped folders are stored
base_path = "cropped_pieces"

# Loop through each model folder inside cropped_pieces
for model_name in os.listdir(base_path):
    model_dir = os.path.join(base_path, model_name)
    if not os.path.isdir(model_dir):
        continue

    # Prepare lists to hold label and pred images
    label_images = []
    pred_images = []

    # Sort and read images
    for filename in sorted(os.listdir(model_dir)):
        filepath = os.path.join(model_dir, filename)
        if filename.startswith("labels"):
            img = cv2.imread(filepath)
            if img is not None:
                label_images.append(img)
        elif filename.startswith("pred"):
            img = cv2.imread(filepath)
            if img is not None:
                pred_images.append(img)

    # Resize all images in a group to the same width and height
    def resize_images_uniformly(images):
        if not images:
            return []
        min_width = min(img.shape[1] for img in images)
        min_height = min(img.shape[0] for img in images)
        resized = [cv2.resize(img, (min_width, min_height)) for img in images]
        return resized

    # Merge label images vertically
    if label_images:
        label_images_resized = resize_images_uniformly(label_images)
        merged_labels = np.vstack(label_images_resized)
        output_path = os.path.join(model_dir, "merged_labels.jpg")
        cv2.imwrite(output_path, merged_labels)
        print(f"Saved merged labels image to {output_path}")

    # Merge pred images vertically
    if pred_images:
        pred_images_resized = resize_images_uniformly(pred_images)
        merged_preds = np.vstack(pred_images_resized)
        output_path = os.path.join(model_dir, "merged_preds.jpg")
        cv2.imwrite(output_path, merged_preds)
        print(f"Saved merged preds image to {output_path}")
