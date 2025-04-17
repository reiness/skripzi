import cv2
import os


def select_rois(image_path, num_rois=6):
    """
    Load an image, allow the user to draw `num_rois` rectangular selections,
    and print out the coordinates for each ROI.

    Args:
        image_path (str): Path to the image file.
        num_rois (int): Number of ROIs to select.
    """
    # Check path
    if not os.path.isfile(image_path):
        print(f"Error: File not found at {image_path}")
        return

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image at {image_path}")
        return

    # Show instructions
    print(f"Loaded image: {image_path} (shape: {img.shape})")
    print(f"Please select {num_rois} regions of interest on the displayed image.")
    print("After drawing each ROI, press ENTER or SPACE to confirm, or 'c' to cancel.")

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)

    rois = []
    for i in range(num_rois):
        # Let user draw ROI; returns (x, y, w, h)
        r = cv2.selectROI("Image", img, showCrosshair=True, fromCenter=False)
        if sum(r) == 0:
            print(f"ROI {i+1} cancelled or zero-sized; skipping.")
        else:
            x, y, w, h = r
            rois.append(r)
            print(f"ROI {i+1}: x={x}, y={y}, width={w}, height={h}")

    cv2.destroyAllWindows()

    # Summary
    print("\nSelected ROIs:")
    for idx, (x, y, w, h) in enumerate(rois, 1):
        print(f"  {idx}: (x={x}, y={y}, w={w}, h={h})")

    return rois


if __name__ == "__main__":
    # Path to your image
    image_path = r"models\YOLOv9s-day\val_batch0_labels.jpg"

    # Select and print ROIs
    select_rois(image_path, num_rois=6)
