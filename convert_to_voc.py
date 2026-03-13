import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image

# -------------------------------------------------------
# Class names from data.yaml (order = class ID)
# -------------------------------------------------------
CLASS_NAMES = [
    'Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100',
    'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30',
    'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70',
    'Speed Limit 80', 'Speed Limit 90', 'Stop'
]

# -------------------------------------------------------
# Paths — update KAGGLE_ROOT to where your dataset is
# -------------------------------------------------------
KAGGLE_ROOT = '/home/ace428/Downloads/traffic_signs_dataset/car'   # <-- CHANGE THIS
OUTPUT_ROOT = 'VOCdevkit/VOC2007'

SPLITS = {
    'train': os.path.join(KAGGLE_ROOT, 'train'),
    'val':   os.path.join(KAGGLE_ROOT, 'valid'),
    'test':  os.path.join(KAGGLE_ROOT, 'test'),
}

# Output dirs
IMG_OUT  = os.path.join(OUTPUT_ROOT, 'JPEGImages')
ANN_OUT  = os.path.join(OUTPUT_ROOT, 'Annotations')
os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(ANN_OUT, exist_ok=True)


def yolo_to_voc_bbox(x_center, y_center, width, height, img_w, img_h):
    """Convert normalized YOLO bbox to absolute VOC (xmin, ymin, xmax, ymax)."""
    xmin = int((x_center - width / 2) * img_w)
    ymin = int((y_center - height / 2) * img_h)
    xmax = int((x_center + width / 2) * img_w)
    ymax = int((y_center + height / 2) * img_h)
    # Clamp to image bounds
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w, xmax)
    ymax = min(img_h, ymax)
    return xmin, ymin, xmax, ymax


def create_xml(image_filename, img_w, img_h, objects):
    """Build a VOC-format XML ElementTree for one image."""
    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = 'JPEGImages'
    ET.SubElement(root, 'filename').text = image_filename

    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text  = str(img_w)
    ET.SubElement(size, 'height').text = str(img_h)
    ET.SubElement(size, 'depth').text  = '3'

    for cls_name, xmin, ymin, xmax, ymax in objects:
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text      = cls_name
        ET.SubElement(obj, 'pose').text      = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    return ET.ElementTree(root)


def convert_split(split_name, split_path):
    img_dir = os.path.join(split_path, 'images')
    lbl_dir = os.path.join(split_path, 'labels')

    image_files = glob.glob(os.path.join(img_dir, '*.jpg')) + \
                  glob.glob(os.path.join(img_dir, '*.jpeg')) + \
                  glob.glob(os.path.join(img_dir, '*.png'))

    converted = 0
    skipped   = 0

    for img_path in image_files:
        img_filename = os.path.basename(img_path)
        stem         = os.path.splitext(img_filename)[0]
        lbl_path     = os.path.join(lbl_dir, stem + '.txt')

        if not os.path.exists(lbl_path):
            print(f"  [SKIP] No label for {img_filename}")
            skipped += 1
            continue

        # Get image dimensions
        with Image.open(img_path) as img:
            img_w, img_h = img.size
            # Convert to jpg if needed and copy to JPEGImages
            jpg_filename = stem + '.jpg'
            out_img_path = os.path.join(IMG_OUT, jpg_filename)
            img.convert('RGB').save(out_img_path, 'JPEG')

        # Parse YOLO labels
        objects = []
        with open(lbl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls_id   = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width    = float(parts[3])
                height   = float(parts[4])

                cls_name = CLASS_NAMES[cls_id]
                xmin, ymin, xmax, ymax = yolo_to_voc_bbox(
                    x_center, y_center, width, height, img_w, img_h
                )
                objects.append((cls_name, xmin, ymin, xmax, ymax))

        # Write XML
        xml_tree = create_xml(jpg_filename, img_w, img_h, objects)
        xml_out  = os.path.join(ANN_OUT, stem + '.xml')
        xml_tree.write(xml_out, encoding='utf-8', xml_declaration=True)

        converted += 1

    print(f"  [{split_name}] Converted: {converted}, Skipped: {skipped}")


if __name__ == '__main__':
    print("Converting YOLO format to VOC XML format...\n")
    for split_name, split_path in SPLITS.items():
        print(f"Processing {split_name} split...")
        convert_split(split_name, split_path)

    print("\nDone! Images -> VOCdevkit/VOC2007/JPEGImages")
    print("       XMLs  -> VOCdevkit/VOC2007/Annotations")
    print("\nNext step: run voc_annotation.py")