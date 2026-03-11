## YOLOv4: Implementation of You Only Look Once Object Detection Model in PyTorch

---

## Table of Contents

1. [Top News](https://www.google.com/search?q=%23top-news)
2. [Related Repositories](https://www.google.com/search?q=%23related-repositories)
3. [Performance](https://www.google.com/search?q=%23performance)
4. [Achievements](https://www.google.com/search?q=%23achievements)
5. [Environment](https://www.google.com/search?q=%23environment)
6. [Download](https://www.google.com/search?q=%23download)
7. [How to Train](https://www.google.com/search?q=%23how-to-train)
8. [How to Predict](https://www.google.com/search?q=%23how-to-predict)
9. [How to Evaluate](https://www.google.com/search?q=%23how-to-evaluate)
10. [Reference](https://www.google.com/search?q=%23reference)

## Top News

**`2023-07`**: Added **Seed settings** to ensure consistent training results across different runs.

**`2022-04`**: Added support for **Multi-GPU training**, calculation of the number of objects for each class, and **heatmaps**.

**`2022-03`**: Major update: modified loss components to balance classification, object, and regression loss ratios. Supported **Step and Cosine learning rate decay**, **Adam and SGD optimizers**, and automatic learning rate adjustment based on batch size. Added **image cropping**.

> Note: The original repository address from the BiliBili video is: [https://github.com/bubbliiiing/yolov4-pytorch/tree/bilibili](https://github.com/bubbliiiing/yolov4-pytorch/tree/bilibili)

**`2021-10`**: Major update: added extensive comments, adjustable parameters, structural modifications, **FPS testing**, video prediction, and batch prediction features.

## Related Repositories

| Model | URL |
| --- | --- |
| YoloV3 | [https://github.com/bubbliiiing/yolo3-pytorch](https://github.com/bubbliiiing/yolo3-pytorch) |
| Efficientnet-Yolo3 | [https://github.com/bubbliiiing/efficientnet-yolo3-pytorch](https://github.com/bubbliiiing/efficientnet-yolo3-pytorch) |
| YoloV4 | [https://github.com/bubbliiiing/yolov4-pytorch](https://github.com/bubbliiiing/yolov4-pytorch) |
| YoloV4-tiny | [https://github.com/bubbliiiing/yolov4-tiny-pytorch](https://github.com/bubbliiiing/yolov4-tiny-pytorch) |
| Mobilenet-Yolov4 | [https://github.com/bubbliiiing/mobilenet-yolov4-pytorch](https://github.com/bubbliiiing/mobilenet-yolov4-pytorch) |
| YoloV5-V5.0 | [https://github.com/bubbliiiing/yolov5-pytorch](https://github.com/bubbliiiing/yolov5-pytorch) |
| YoloV5-V6.1 | [https://github.com/bubbliiiing/yolov5-v6.1-pytorch](https://github.com/bubbliiiing/yolov5-v6.1-pytorch) |
| YoloX | [https://github.com/bubbliiiing/yolox-pytorch](https://github.com/bubbliiiing/yolox-pytorch) |
| YoloV7 | [https://github.com/bubbliiiing/yolov7-pytorch](https://github.com/bubbliiiing/yolov7-pytorch) |
| YoloV7-tiny | [https://github.com/bubbliiiing/yolov7-tiny-pytorch](https://github.com/bubbliiiing/yolov7-tiny-pytorch) |

## Performance

| Training Dataset | Weight File | Test Dataset | Input Size | mAP 0.5:0.95 | mAP 0.5 |
| --- | --- | --- | --- | --- | --- |
| VOC07+12+COCO | [yolo4_voc_weights.pth](https://github.com/bubbliiiing/yolov4-pytorch/releases/download/v1.0/yolo4_voc_weights.pth) | VOC-Test07 | 416x416 | - | 89.0 |
| COCO-Train2017 | [yolo4_weights.pth](https://github.com/bubbliiiing/yolov4-pytorch/releases/download/v1.0/yolo4_weights.pth) | COCO-Val2017 | 416x416 | 46.1 | 70.2 |

## Achievements

* [x] **Backbone**: DarkNet53 => CSPDarkNet53
* [x] **Neck**: SPP, PAN
* [x] **Training Tricks**: Mosaic Data Augmentation, Label Smoothing, CIOU, Cosine Annealing LR Decay
* [x] **Activation**: Mish

## Training Steps

### a. Training on VOC07+12

1. **Preparation**: Download the VOC07+12 dataset, unzip it, and place it in the root directory.
2. **Processing**: Set `annotation_mode=2` in `voc_annotation.py` and run it to generate `2007_train.txt` and `2007_val.txt`.
3. **Start Training**: Run `train.py`. The default parameters are set for VOC.
4. **Prediction**: Modify `model_path` (pointing to the weight file in `logs/`) and `classes_path` (pointing to the class txt) in `yolo.py`, then run `predict.py`.

### b. Training Your Own Dataset

1. **Preparation**: Place annotation files in `VOCdevkit/VOC2007/Annotations` and images in `VOCdevkit/VOC2007/JPEGImages`.
2. **Processing**: Create a `cls_classes.txt` listing your categories. Modify `classes_path` in `voc_annotation.py` to point to it and run the script to generate training txt files.
3. **Start Training**: Ensure `classes_path` in `train.py` matches your custom class file. Run `train.py`. Weights will save to the `logs/` folder.
4. **Prediction**: Update `model_path` and `classes_path` in `yolo.py` before running `predict.py`.

## Prediction Steps

### a. Using Pre-trained Weights

1. Download `yolo_weights.pth`, place it in `model_data`, and run `predict.py`. Input an image path like `img/street.jpg`.
2. FPS and video detection can be enabled in `predict.py`.

### b. Using Your Own Weights

1. Follow the training steps.
2. In `yolo.py`, update the `_defaults` dictionary to match your trained weights and classes.
3. Run `predict.py`.

## Evaluation Steps

### a. Evaluating VOC07+12

1. Update `model_path` and `classes_path` in `yolo.py`.
2. Run `get_map.py`. Results are saved in `map_out/`.

### b. Evaluating Your Own Dataset

1. Use the test set split generated by `voc_annotation.py` (default train+val/test ratio is 9:1).
2. Update `classes_path` in `get_map.py` and both `model_path` and `classes_path` in `yolo.py`.
3. Run `get_map.py`.

## Reference

* [https://github.com/qqwweee/keras-yolo3/](https://github.com/qqwweee/keras-yolo3/)
* [https://github.com/Cartucho/mAP](https://github.com/Cartucho/mAP)
* [https://github.com/Ma-Dan/keras-yolo4](https://github.com/Ma-Dan/keras-yolo4)

---
