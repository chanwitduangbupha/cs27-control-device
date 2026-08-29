# ชุด Train Model จำแนก GREEN และ RED

ชุดนี้ใช้สำหรับจำแนกบัตรสัตว์ออกเป็น 2 คลาส

- `GREEN` = สัตว์บก = +1 คะแนน
- `RED` = สัตว์น้ำ = 0 คะแนน

เทคโนโลยีที่ใช้

- Python
- Ultralytics YOLOv8 Classification
- OpenCV
- Webcam

## โครงสร้างโฟลเดอร์

```text
yolov8_green_red_package/
├── dataset/
│   ├── train/
│   │   ├── GREEN/
│   │   └── RED/
│   ├── val/
│   │   ├── GREEN/
│   │   └── RED/
│   └── test/
│       ├── GREEN/
│       └── RED/
├── raw/
│   ├── GREEN/
│   └── RED/
├── train_model.py
├── predict_image.py
├── webcam_detect.py
├── split_dataset.py
└── requirements.txt
```

## 1. ติดตั้งไลบรารี

```bash
pip install -r requirements.txt
```

## 2. เพิ่มภาพสำหรับ Train

นำภาพสัตว์บกใส่ใน

```text
raw/GREEN/
```

นำภาพสัตว์น้ำใส่ใน

```text
raw/RED/
```

ควรถ่ายบัตรแต่ละใบหลายรูป โดยเปลี่ยน

- ระยะใกล้และไกล
- มุมเอียง
- แสงสว่าง
- พื้นหลัง
- ตำแหน่งของบัตร
- กล้องจริงที่จะใช้ในงาน

แนะนำอย่างน้อย 100–200 ภาพต่อคลาส  
สำหรับใช้งานจริงควรมี 300–500 ภาพต่อคลาส

## 3. แบ่งข้อมูลอัตโนมัติ

```bash
python split_dataset.py
```

อัตราการแบ่งคือ

- Train 70%
- Validation 20%
- Test 10%

## 4. Train โมเดล

```bash
python train_model.py
```

ไฟล์โมเดลที่ดีที่สุดจะอยู่ที่

```text
runs/green_red_classifier/weights/best.pt
```

## 5. ทดสอบจากไฟล์ภาพ

```bash
python predict_image.py
```

## 6. ทดสอบด้วย Webcam

```bash
python webcam_detect.py
```

วางบัตรไว้ในกรอบกลางหน้าจอ และกด `Q` เพื่อปิดโปรแกรม

## ข้อสำคัญ

ภาพตัวอย่างในแพ็กเกจมีจำนวนน้อย ใช้สำหรับทดสอบโครงสร้างโปรแกรมเท่านั้น  
ก่อนนำไปใช้จริงควรเพิ่มภาพจากกล้อง Webcam ตัวจริงให้เพียงพอ มิฉะนั้นโมเดลอาจจำภาพตัวอย่างมากกว่าการจำแนกสีและลักษณะของบัตร
