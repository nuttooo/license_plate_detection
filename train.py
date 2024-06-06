import os
from ultralytics import YOLO
from dotenv import load_dotenv

# โหลดค่าคอนฟิกูเรชันจากไฟล์ .env
load_dotenv()

# ดึงค่าคอนฟิกูเรชันจาก environment variables
dataset_path = os.getenv('DATASET_PATH', 'dataset')
epochs = int(os.getenv('EPOCHS', 100))
img_size = int(os.getenv('IMG_SIZE', 640))

# โหลดโมเดล YOLOv8 ที่เตรียมเทรน
model = YOLO('yolov8n.pt')

# เทรนโมเดล
path = '/Users/kamonwat.dev/Devonix/Work.dev/license_plate_detection/thai.v2i.yolov8/data.yaml'
results = model.train(data=path, epochs=epochs, imgsz=img_size)

# บันทึกโมเดล
model.export(format='onnx')
