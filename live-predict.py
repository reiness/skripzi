from ultralytics import RTDETR

model = RTDETR("C:/Users/redoks/Documents/skripzii/products6k/RTDETR/RTDETR-X/batch16_lr0.001/weights/best.pt")

model.predict("C:/Users/redoks/Documents/skripzii/data/test_collections/test_raw/bicycle_test.mp4", save=True, imgsz=640, conf=0.25, device = 'cpu')