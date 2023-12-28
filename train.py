from ultralytics import YOLO

model = YOLO("yolov8n.pt")

if __name__ == '__main__':
    results = model.train(data="datasets/data.yaml", epochs=500, device=0)
    results = model.val()
