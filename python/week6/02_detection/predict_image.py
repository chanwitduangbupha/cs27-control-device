from pathlib import Path
from ultralytics import YOLO


MODEL_PATH = Path("runs/animal_classifier_result/weights/best.pt")
IMAGE_PATH = Path("dataset/test/WATER/green_05.jpg")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ไม่พบโมเดล {MODEL_PATH}\nกรุณารัน train_model.py ก่อน"
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"ไม่พบภาพทดสอบ {IMAGE_PATH}")

    model = YOLO(str(MODEL_PATH))
    results = model.predict(source=str(IMAGE_PATH), imgsz=224, verbose=False)

    result = results[0]
    top1 = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    class_name = result.names[top1]

    print(f"ผลการทำนาย: {class_name}")
    print(f"ความมั่นใจ: {confidence:.2%}")


if __name__ == "__main__":
    main()
