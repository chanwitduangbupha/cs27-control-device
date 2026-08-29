from ultralytics import YOLO


def main():
    # โมเดล Classification เหมาะกับการจำแนก GREEN และ RED
    model = YOLO("yolov8n-cls.pt")

    model.train(
        data="dataset",
        epochs=50,
        imgsz=224,
        batch=16,
        patience=10,
        device="cpu",  # เปลี่ยนเป็น 0 เมื่อใช้ NVIDIA GPU
        project="runs",
        name="animal_classifier_result",
        plots=True,
    )


if __name__ == "__main__":
    main()
