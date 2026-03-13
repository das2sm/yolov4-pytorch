Training:

YOLOv4 on 15-class traffic signs dataset
~80%+ mAP@0.5 with Adam optimizer as the key fix

Deployment pipeline:

PyTorch → ONNX → RKNN with INT8 quantization
Deployed on OrangePi 5 Ultra RK3588 NPU

Inference:

3-core NPU inference pool
Vectorized numpy post-processing
Threaded camera capture
60-70 FPS on hardware that cost a fraction of a GPU
