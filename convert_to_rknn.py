from rknn.api import RKNN

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
ONNX_MODEL      = 'model_data/models.onnx'
RKNN_MODEL      = 'model_data/models.rknn'
CALIBRATION_TXT = 'calibration_images.txt'

# -------------------------------------------------------
# Target platform
# -------------------------------------------------------
PLATFORM = 'rk3588'

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # -------------------------------------------------------
    # Configure model
    # -------------------------------------------------------
    print('Configuring RKNN model...')
    rknn.config(
        mean_values=[[0, 0, 0]],        # YOLOv4 normalizes in preprocess, so mean=0
        std_values=[[255, 255, 255]],   # divides by 255
        target_platform=PLATFORM,
        quantized_algorithm='normal',
        quantized_method='channel',
    )

    # -------------------------------------------------------
    # Load ONNX model
    # -------------------------------------------------------
    print('Loading ONNX model...')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Failed to load ONNX model')
        exit(ret)
    print('ONNX model loaded successfully')

    # -------------------------------------------------------
    # Build RKNN model with INT8 quantization
    # -------------------------------------------------------
    print('Building RKNN model with INT8 quantization...')
    ret = rknn.build(
        do_quantization=True,
        dataset=CALIBRATION_TXT
    )
    if ret != 0:
        print('Failed to build RKNN model')
        exit(ret)
    print('RKNN model built successfully')

    # -------------------------------------------------------
    # Export RKNN model
    # -------------------------------------------------------
    print('Exporting RKNN model...')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Failed to export RKNN model')
        exit(ret)
    print(f'RKNN model saved to {RKNN_MODEL}')

    rknn.release()
    print('Done!')