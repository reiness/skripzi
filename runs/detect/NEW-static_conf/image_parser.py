import cv2
import os

# Define the ROIs (Regions of Interest)
rois = [
    (3, 324, 477, 268),
    (4, 5, 476, 270),
    (4, 962, 475, 271),
    (483, 8, 475, 263),
    (482, 321, 478, 274),
    (1, 643, 479, 271)
]

# Base directory containing model folders
models_base_path = "models"

# Iterate through each model folder
for model_name in os.listdir(models_base_path):
    model_dir = os.path.join(models_base_path, model_name)
    if not os.path.isdir(model_dir):
        continue

    # Iterate through each image in the model folder
    for image_filename in os.listdir(model_dir):
        image_path = os.path.join(model_dir, image_filename)

        # Determine prefix based on filename
        if "labels" in image_filename:
            prefix = "labels"
        elif "pred" in image_filename:
            prefix = "pred"
        else:
            prefix = "crop"

        # Output directory for crops
        output_dir = os.path.join("cropped_pieces", model_name)
        os.makedirs(output_dir, exist_ok=True)

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Failed to load image at path: {image_path}")
            continue

        # Crop and save each ROI
        for idx, (x, y, w, h) in enumerate(rois, start=1):
            cropped = img[y:y+h, x:x+w]
            output_path = os.path.join(output_dir, f"{prefix}_{idx}.jpg")
            cv2.imwrite(output_path, cropped)
            print(f"Saved crop {idx} to {output_path}")
