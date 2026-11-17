import os
from PIL import Image

DATASET_ROOT = os.path.expanduser("~/Desktop/VisDrone")

SETS = [
    "VisDrone2019-DET-train",
    "VisDrone2019-DET-val",
]

for split in SETS:
    split_path = os.path.join(DATASET_ROOT, split)
    image_dir = os.path.join(split_path, "images")
    annotation_dir = os.path.join(split_path, "annotations")
    label_dir = os.path.join(split_path, "labels")

    os.makedirs(label_dir, exist_ok=True)

    for annotation_file in os.listdir(annotation_dir):
        if not annotation_file.endswith(".txt"):
            continue

        image_name = os.path.splitext(annotation_file)[0]

        # Find corresponding image
        image_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(image_dir, image_name + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            print(f"Image not found: {image_name}")
            continue

        # Get actual image dimensions
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        output_lines = []

        annotation_path = os.path.join(annotation_dir, annotation_file)

        with open(annotation_path, "r") as f:
            for line in f:
                values = line.strip().split(",")

                if len(values) != 8:
                    continue

                x, y, w, h, score, category, truncation, occlusion = map(
                    int, values
                )

                # Ignore regions and invalid categories
                if category in [0, 11]:
                    continue

                # VisDrone classes 1-10 → YOLO classes 0-9
                class_id = category - 1

                # Convert top-left format to center format
                x_center = x + w / 2
                y_center = y + h / 2

                # Normalize coordinates
                x_center /= image_width
                y_center /= image_height
                w /= image_width
                h /= image_height

                output_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} "
                    f"{w:.6f} {h:.6f}\n"
                )

        # Save YOLO label
        label_path = os.path.join(
            label_dir, image_name + ".txt"
        )

        with open(label_path, "w") as f:
            f.writelines(output_lines)

    print(f"Finished: {split}")
