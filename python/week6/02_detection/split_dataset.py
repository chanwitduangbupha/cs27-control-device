from pathlib import Path
import random
import shutil


RAW_DIR = Path("raw")
DATASET_DIR = Path("dataset")
CLASSES = ["GREEN", "RED"]
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
SEED = 42


def copy_files(files, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        shutil.copy2(file_path, destination / file_path.name)


def main():
    random.seed(SEED)

    for class_name in CLASSES:
        source_dir = RAW_DIR / class_name
        files = [
            p for p in source_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]

        if len(files) < 10:
            print(
                f"คำเตือน: {class_name} มีเพียง {len(files)} ภาพ "
                "ควรมีอย่างน้อย 50–100 ภาพต่อคลาส"
            )

        random.shuffle(files)

        total = len(files)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        copy_files(train_files, DATASET_DIR / "train" / class_name)
        copy_files(val_files, DATASET_DIR / "val" / class_name)
        copy_files(test_files, DATASET_DIR / "test" / class_name)

        print(
            f"{class_name}: train={len(train_files)}, "
            f"val={len(val_files)}, test={len(test_files)}"
        )


if __name__ == "__main__":
    main()
