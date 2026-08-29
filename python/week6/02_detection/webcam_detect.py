import sys
import time
from pathlib import Path

import cv2
import serial
import serial.tools.list_ports
from ultralytics import YOLO

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QFontDatabase, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QMessageBox,
    QFrame,
    QSizePolicy,
    QSpacerItem,
)


# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "runs/animal_classifier_result/weights/best.pt"
LOGO_PATH = "logo_transparent.png"

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

BAUD_RATE = 9600
CONFIDENCE_THRESHOLD = 0.90
STABLE_FRAMES = 6

# โมเดลนี้ใช้ 2 Class: LAND, WATER
VALID_CLASSES = ("NONE","LAND", "WATER")

# YOLO ทำงานทุก N เฟรม ช่วยให้เครื่องทั่วไปไม่หน่วง
DETECT_EVERY_N_FRAMES = 2

# ROI อิงตามสัดส่วนภาพจากกล้อง
ROI_WIDTH_RATIO = 0.55
ROI_HEIGHT_RATIO = 0.48

# ฟอนต์หลักของโปรแกรม
FONT_FAMILY = "Prompt"


class YoloCardController(QWidget):
    @staticmethod
    def get_ui_font_family():
        """ใช้ Prompt ถ้ามีในเครื่อง ถ้าไม่มีให้ fallback เป็นฟอนต์ไทยของ Windows"""
        families = {name.lower(): name for name in QFontDatabase.families()}
        for candidate in ("Prompt", "Leelawadee UI", "Tahoma", "Segoe UI"):
            if candidate.lower() in families:
                return families[candidate.lower()]
        return QApplication.font().family()

    def __init__(self):
        super().__init__()

        self.model = None
        self.camera = None
        self.serial_port = None

        self.frame_counter = 0
        self.stable_name = ""
        self.stable_count = 0
        self.last_sent = ""

        self.current_command = "UNKNOWN"
        self.current_confidence = 0.0
        self.last_frame = None

        self.logo_original = QPixmap(LOGO_PATH) if Path(LOGO_PATH).exists() else QPixmap()

        self.ui_font_family = self.get_ui_font_family()

        self.init_model()
        self.init_ui()
        self.load_com_ports()
        self.start_camera()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ========================================================
    # MODEL
    # ========================================================
    def init_model(self):
        if not Path(MODEL_PATH).exists():
            QMessageBox.critical(
                self,
                "Model Error",
                f"ไม่พบโมเดล:\n{Path(MODEL_PATH).resolve()}",
            )
            return

        try:
            self.model = YOLO(MODEL_PATH)

            model_names = {
                str(name).upper()
                for name in self.model.names.values()
            }

            expected = set(VALID_CLASSES)

            if not expected.issubset(model_names):
                QMessageBox.warning(
                    self,
                    "Class Warning",
                    "โมเดลควรมี Class: LAND, WATER\n"
                    f"Class ที่พบ: {sorted(model_names)}",
                )

        except Exception as error:
            QMessageBox.critical(self, "Model Error", str(error))

    # ========================================================
    # UI
    # ========================================================
    def init_ui(self):
        self.setWindowTitle("ระบบจำแนกบัตรภาพการ์ตูนด้วย YOLO - โดย CS-MCRU")

        screen = QApplication.primaryScreen().availableGeometry()
        start_width = min(1320, int(screen.width() * 0.90))
        start_height = min(900, int(screen.height() * 0.90))

        self.resize(start_width, start_height)
        self.setMinimumSize(640, 480)

        self.setStyleSheet(
            """
            QWidget {
                background-color: #f2f4f7;
                font-family: "Prompt", "Leelawadee UI", "Tahoma", "Segoe UI", sans-serif;
            }

            QFrame#headerPanel,
            QFrame#resultPanel,
            QFrame#bottomPanel {
                background-color: white;
                border: 1px solid #d8dee8;
                border-radius: 10px;
            }

            QLabel#titleLabel {
                color: #172033;
                font-weight: 700;
            }

            QLabel#cameraLabel {
                background-color: #101827;
                color: white;
                border: 2px solid #283548;
                border-radius: 10px;
            }

            QLabel#commandLabel {
                font-weight: 800;
                border-radius: 8px;
            }

            QLabel#confidenceLabel {
                color: #4b5563;
            }

            QLabel#statusLabel {
                font-weight: 700;
            }

            QComboBox {
                min-height: 38px;
                padding: 3px 10px;
                border: 1px solid #9ca3af;
                border-radius: 7px;
                background-color: white;
            }

            QPushButton {
                min-height: 40px;
                padding: 4px 16px;
                border: none;
                border-radius: 7px;
                color: white;
                font-weight: 700;
            }

            QPushButton#refreshButton {
                background-color: #667085;
            }

            QPushButton#connectButton {
                background-color: #16a34a;
            }

            QPushButton#disconnectButton {
                background-color: #dc2626;
            }

            QPushButton:hover:!disabled {
                border: 2px solid rgba(255, 255, 255, 0.65);
            }

            QPushButton:disabled {
                background-color: #d1d5db;
                color: #8b95a5;
            }
            """
        )

        # ---------------- Header ----------------
        self.header_panel = QFrame()
        self.header_panel.setObjectName("headerPanel")

        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title_label = QLabel("ระบบจำแนกบัตรภาพการ์ตูนด้วย YOLO : โดย CS-MCRU")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setWordWrap(True)

        header_layout = QHBoxLayout(self.header_panel)
        header_layout.setContentsMargins(14, 8, 14, 8)
        header_layout.setSpacing(12)
        header_layout.addWidget(self.logo_label, 0)
        header_layout.addWidget(self.title_label, 1)
        self.header_spacer = QLabel()
        header_layout.addWidget(self.header_spacer, 0)

        # ---------------- Camera ----------------
        self.camera_hint_label = QLabel("กรุณาวางบัตรภาพการ์ตูนภายในกรอบสีเหลือง")
        self.camera_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_hint_label.setStyleSheet(
            "color: #b58900; font-weight: 700; background: transparent;"
        )

        self.camera_label = QLabel("กำลังเปิด Webcam...")
        self.camera_label.setObjectName("cameraLabel")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(480, 270)
        self.camera_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        # ---------------- Result panel ----------------
        self.result_panel = QFrame()
        self.result_panel.setObjectName("resultPanel")

        self.command_label = QLabel("UNKNOWN")
        self.command_label.setObjectName("commandLabel")
        self.command_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.confidence_label = QLabel("Confidence: 0.00")
        self.confidence_label.setObjectName("confidenceLabel")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # ป้องกันข้อความผลลัพธ์ถูกบีบเมื่อหน้าต่างมีขนาดเล็ก
        self.command_label.setMinimumHeight(42)
        self.confidence_label.setMinimumHeight(28)
        self.command_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.MinimumExpanding,
        )

        result_layout = QVBoxLayout(self.result_panel)
        result_layout.setContentsMargins(10, 8, 10, 8)
        result_layout.setSpacing(4)
        result_layout.addWidget(self.command_label)
        result_layout.addWidget(self.confidence_label)

        # ---------------- Bottom panel ----------------
        self.bottom_panel = QFrame()
        self.bottom_panel.setObjectName("bottomPanel")

        self.port_combo = QComboBox()
        self.port_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self.refresh_button = QPushButton("ค้นหา Port ใหม่")
        self.refresh_button.setObjectName("refreshButton")

        self.connect_button = QPushButton("เชื่อมต่อ")
        self.connect_button.setObjectName("connectButton")

        self.disconnect_button = QPushButton("ตัดการเชื่อมต่อ")
        self.disconnect_button.setObjectName("disconnectButton")
        self.disconnect_button.setEnabled(False)

        self.connection_status = QLabel("สถานะ: ยังไม่ได้เชื่อมต่อ")
        self.connection_status.setObjectName("statusLabel")
        self.connection_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.connection_status.setStyleSheet("color: #dc2626;")

        self.refresh_button.clicked.connect(self.load_com_ports)
        self.connect_button.clicked.connect(self.connect_serial)
        self.disconnect_button.clicked.connect(self.disconnect_serial)

        self.control_layout = QGridLayout(self.bottom_panel)
        self.control_layout.setContentsMargins(14, 12, 14, 12)
        self.control_layout.setHorizontalSpacing(10)
        self.control_layout.setVerticalSpacing(8)

        # ---------------- Main ----------------
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 8, 10, 8)
        self.main_layout.setSpacing(8)

        self.main_layout.addWidget(self.header_panel, 0)
        self.main_layout.addWidget(self.camera_hint_label, 0)
        self.main_layout.addWidget(self.camera_label, 1)
        self.main_layout.addWidget(self.result_panel, 0)
        self.main_layout.addWidget(self.bottom_panel, 0)

        self.apply_responsive_layout()

    def clear_grid(self):
        while self.control_layout.count():
            item = self.control_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(self.bottom_panel)

    def apply_responsive_layout(self):
        width = max(self.width(), 720)
        height = max(self.height(), 560)

        compact = width < 1100
        very_compact = width < 820

        # ขนาด Header และฟอนต์
        logo_size = max(42, min(90, int(width * 0.065)))
        title_size = max(14, min(28, int(width * 0.020)))
        command_size = max(16, min(30, int(width * 0.023)))
        normal_size = max(11, min(16, int(width * 0.012)))
        button_size = max(11, min(15, int(width * 0.0115)))

        self.logo_label.setFixedSize(logo_size, logo_size)
        self.header_spacer.setFixedWidth(logo_size)

        if not self.logo_original.isNull():
            self.logo_label.setPixmap(
                self.logo_original.scaled(
                    logo_size,
                    logo_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        title_font = QFont(self.ui_font_family, title_size)
        title_font.setBold(True)
        self.title_label.setFont(title_font)

        command_font = QFont(self.ui_font_family, command_size)
        command_font.setBold(True)
        self.command_label.setFont(command_font)

        confidence_font = QFont(self.ui_font_family, normal_size)
        self.confidence_label.setFont(confidence_font)
        self.connection_status.setFont(confidence_font)
        self.port_combo.setFont(confidence_font)
        self.camera_hint_label.setFont(QFont(self.ui_font_family, normal_size, QFont.Weight.Bold))

        for button in (
            self.refresh_button,
            self.connect_button,
            self.disconnect_button,
        ):
            button.setFont(QFont(self.ui_font_family, button_size, QFont.Weight.Bold))

        # ความสูงตามขนาดหน้าต่าง
        header_height = max(62, min(115, int(height * 0.13)))
        result_height = max(86, min(125, int(height * 0.14)))
        bottom_height = max(105, min(165, int(height * 0.18)))

        self.header_panel.setMaximumHeight(header_height)
        self.result_panel.setMinimumHeight(result_height)
        self.result_panel.setMaximumHeight(result_height)
        self.bottom_panel.setMinimumHeight(bottom_height)

        self.clear_grid()

        if very_compact:
            # หน้าจอแคบมาก: พอร์ต 1 แถว ปุ่ม 2 แถว
            self.control_layout.addWidget(self.port_combo, 0, 0, 1, 2)
            self.control_layout.addWidget(self.refresh_button, 1, 0, 1, 2)
            self.control_layout.addWidget(self.connect_button, 2, 0)
            self.control_layout.addWidget(self.disconnect_button, 2, 1)
            self.control_layout.addWidget(self.connection_status, 3, 0, 1, 2)
            self.control_layout.setColumnStretch(0, 1)
            self.control_layout.setColumnStretch(1, 1)

        elif compact:
            # หน้าจอกลาง: พอร์ต+Refresh และ Connect+Disconnect
            self.control_layout.addWidget(self.port_combo, 0, 0)
            self.control_layout.addWidget(self.refresh_button, 0, 1)
            self.control_layout.addWidget(self.connect_button, 1, 0)
            self.control_layout.addWidget(self.disconnect_button, 1, 1)
            self.control_layout.addWidget(self.connection_status, 2, 0, 1, 2)
            self.control_layout.setColumnStretch(0, 1)
            self.control_layout.setColumnStretch(1, 1)

        else:
            # หน้าจอกว้าง: ทุกอย่างในแถวเดียวและกึ่งกลาง
            self.control_layout.addItem(
                QSpacerItem(
                    10,
                    10,
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Minimum,
                ),
                0,
                0,
            )
            self.control_layout.addWidget(self.port_combo, 0, 1)
            self.control_layout.addWidget(self.refresh_button, 0, 2)
            self.control_layout.addWidget(self.connect_button, 0, 3)
            self.control_layout.addWidget(self.disconnect_button, 0, 4)
            self.control_layout.addItem(
                QSpacerItem(
                    10,
                    10,
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Minimum,
                ),
                0,
                5,
            )
            self.control_layout.addWidget(self.connection_status, 1, 0, 1, 6)

            self.control_layout.setColumnStretch(0, 1)
            self.control_layout.setColumnStretch(1, 2)
            self.control_layout.setColumnStretch(2, 1)
            self.control_layout.setColumnStretch(3, 1)
            self.control_layout.setColumnStretch(4, 1)
            self.control_layout.setColumnStretch(5, 1)

    # ========================================================
    # CAMERA
    # ========================================================
    def start_camera(self):
        self.camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        if not self.camera.isOpened():
            QMessageBox.critical(self, "Webcam Error", "เปิด Webcam ไม่สำเร็จ")
            return

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def update_frame(self):
        if self.camera is None or not self.camera.isOpened():
            return

        ok, frame = self.camera.read()
        if not ok:
            return

        self.frame_counter += 1
        frame_height, frame_width = frame.shape[:2]

        roi_width = int(frame_width * ROI_WIDTH_RATIO)
        roi_height = int(frame_height * ROI_HEIGHT_RATIO)

        x1 = (frame_width - roi_width) // 2
        y1 = (frame_height - roi_height) // 2
        x2 = x1 + roi_width
        y2 = y1 + roi_height

        roi = frame[y1:y2, x1:x2].copy()

        # ตรวจจับตามช่วงเฟรม เพื่อให้ภาพลื่น
        if self.frame_counter % DETECT_EVERY_N_FRAMES == 0:
            command = "UNKNOWN"
            confidence = 0.0

            if self.model is not None:
                result = self.model.predict(
                    roi,
                    imgsz=224,
                    verbose=False,
                )[0]

                class_index = int(result.probs.top1)
                confidence = float(result.probs.top1conf)
                class_name = result.names[class_index].upper()

                if (
                    confidence >= CONFIDENCE_THRESHOLD
                    and class_name in ("LAND", "WATER")
                ):
                    command = class_name
                else:
                    command = "UNKNOWN"

            self.current_command = command
            self.current_confidence = confidence
            self.process_stable_command(command)

        # ขนาดเส้นและตัวอักษรอิงตามภาพจริง
        line_thickness = max(2, int(frame_width / 550))
        font_scale = max(0.55, min(1.0, frame_width / 1450))
        text_thickness = max(1, int(frame_width / 700))

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            line_thickness,
        )

        cv2.putText(
            frame,
            "Place card inside the box",
            (x1, max(y1 - 14, 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            text_thickness + 1,
            cv2.LINE_AA,
        )

        self.update_command_display(
            self.current_command,
            self.current_confidence,
        )

        self.last_frame = frame.copy()
        self.display_frame(frame)

    def process_stable_command(self, command):
        stable_value = command

        if stable_value == self.stable_name:
            self.stable_count += 1
        else:
            self.stable_name = stable_value
            self.stable_count = 1

        if (
            self.stable_count >= STABLE_FRAMES
            and stable_value != self.last_sent
        ):
            # LAND  -> GREEN
            # WATER -> RED
            # UNKNOWN -> OFF
            if stable_value == "LAND":
                self.send_command("GREEN")
            elif stable_value == "WATER":
                self.send_command("RED")
            else:
                self.send_command("OFF")

            self.last_sent = stable_value

    def display_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape

        image = QImage(
            rgb_frame.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()

        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.camera_label.setPixmap(pixmap)

    # ========================================================
    # COMMAND DISPLAY
    # ========================================================
    def update_command_display(self, command, confidence):
        if command == "LAND":
            display_text = "+1 คะแนน"
            background = "#16a34a"
            foreground = "white"

        elif command == "WATER":
            display_text = "0 คะแนน"
            background = "#c70202"
            foreground = "white"

        else:
            display_text = "ไม่พบข้อมูล"
            background = "#e5e7eb"
            foreground = "#111827"

        self.command_label.setText(display_text)
        self.confidence_label.setText(
            f"confidence : {confidence:.2f}"
        )

        padding = max(
            6,
            min(14, int(self.width() * 0.008))
        )

        self.command_label.setStyleSheet(
            f"""
            font-weight: 800;
            padding: {padding}px;
            border-radius: 8px;
            background-color: {background};
            color: {foreground};
            """
        )

    # ========================================================
    # COM PORT
    # ========================================================
    def load_com_ports(self):
        current_port = self.port_combo.currentData()
        self.port_combo.clear()

        ports = list(serial.tools.list_ports.comports())

        for port in ports:
            self.port_combo.addItem(
                f"{port.device} - {port.description}",
                port.device,
            )

        if self.port_combo.count() == 0:
            self.port_combo.addItem("- n/a", None)
            return

        for index in range(self.port_combo.count()):
            if self.port_combo.itemData(index) == "COM3":
                self.port_combo.setCurrentIndex(index)
                return

        if current_port:
            for index in range(self.port_combo.count()):
                if self.port_combo.itemData(index) == current_port:
                    self.port_combo.setCurrentIndex(index)
                    break

    def connect_serial(self):
        port_name = self.port_combo.currentData()

        if not port_name:
            QMessageBox.warning(
                self,
                "COM Port",
                "ไม่พบ COM Port กรุณาตรวจสอบสาย USB และ Driver",
            )
            return

        try:
            self.serial_port = serial.Serial(
                port=port_name,
                baudrate=BAUD_RATE,
                timeout=1,
                write_timeout=1,
            )

            time.sleep(1.5)

            self.connection_status.setText(
                f"สถานะ: เชื่อมต่อ {port_name} สำเร็จ"
            )
            self.connection_status.setStyleSheet("color: #16a34a;")

            self.port_combo.setEnabled(False)
            self.refresh_button.setEnabled(False)
            self.connect_button.setEnabled(False)
            self.disconnect_button.setEnabled(True)

        except serial.SerialException as error:
            QMessageBox.critical(self, "Serial Error", str(error))
            self.serial_port = None

    def disconnect_serial(self):
        if self.serial_port is not None and self.serial_port.is_open:
            self.serial_port.close()

        self.serial_port = None
        self.last_sent = ""

        self.connection_status.setText("สถานะ: ตัดการเชื่อมต่อแล้ว")
        self.connection_status.setStyleSheet("color: #dc2626;")

        self.port_combo.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)

    def send_command(self, command):
        if self.serial_port is None or not self.serial_port.is_open:
            print(f"ตรวจพบ {command} แต่ยังไม่ได้เชื่อมต่อ COM Port")
            return

        try:
            self.serial_port.write((command + "\n").encode("utf-8"))
            self.serial_port.flush()
            print("Sent:", command)

        except serial.SerialException as error:
            QMessageBox.critical(self, "Serial Error", str(error))
            self.disconnect_serial()

    # ========================================================
    # RESPONSIVE EVENTS
    # ========================================================
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.apply_responsive_layout()

        # ปรับภาพกล้องเดิมทันทีระหว่างย่อ/ขยาย
        if self.last_frame is not None:
            self.display_frame(self.last_frame)

    def closeEvent(self, event):
        self.timer.stop()

        if self.camera is not None:
            self.camera.release()

        if self.serial_port is not None and self.serial_port.is_open:
            self.serial_port.close()

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # ตั้งฟอนต์หลักทั้งโปรแกรมเป็น Prompt ถ้ามีในระบบ
    available_fonts = {name.lower(): name for name in QFontDatabase.families()}
    selected_font = None
    for candidate in ("Prompt", "Leelawadee UI", "Tahoma", "Segoe UI"):
        if candidate.lower() in available_fonts:
            selected_font = available_fonts[candidate.lower()]
            break

    if selected_font:
        app.setFont(QFont(selected_font, 11))

    window = YoloCardController()
    window.show()

    sys.exit(app.exec())