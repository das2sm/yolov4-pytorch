#-----------------------------------------------------------------------#
#   predict.py consolidates single image prediction, camera detection,
#   FPS testing, and folder batch detection into one file.
#   Change the 'mode' variable to switch between modes.
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO, YOLO_ONNX

if __name__ == "__main__":
    #----------------------------------------------------------------------------------------------------------#
    #   mode specifies which function to run:
    #   'predict'           Single image prediction. See comments below for saving images, cropping objects, etc.
    #   'video'             Video detection. Can use a webcam or a video file. See comments below.
    #   'fps'               FPS test using img/street.jpg. See comments below.
    #   'dir_predict'       Batch detect all images in a folder and save results. See comments below.
    #   'heatmap'           Visualize prediction results as a heatmap. See comments below.
    #   'export_onnx'       Export the model to ONNX format. Requires PyTorch >= 1.7.1.
    #   'predict_onnx'      Run inference using an exported ONNX model.
    #                       Relevant parameters are near line 416 in yolo.py under YOLO_ONNX.
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    #-------------------------------------------------------------------------#
    #   crop        Whether to crop detected objects out of the image after prediction.
    #   count       Whether to count the number of detected objects.
    #   crop and count are only active when mode='predict'.
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          Path to the video file. Set to 0 to use the webcam.
    #                       To detect a video file, set e.g. video_path = "xxx.mp4".
    #   video_save_path     Path to save the output video. Set to "" to not save.
    #                       To save, set e.g. video_save_path = "yyy.mp4".
    #   video_fps           FPS for the saved output video.
    #
    #   video_path, video_save_path, and video_fps are only active when mode='video'.
    #   To complete the save properly, either press Ctrl+C to exit or let the video run to the last frame.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       Number of times to run detection for FPS measurement.
    #                       Higher values give more accurate FPS readings.
    #   fps_image_path      Path to the image used for FPS testing.
    #
    #   test_interval and fps_image_path are only active when mode='fps'.
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     Folder containing images to detect.
    #   dir_save_path       Folder to save detection results.
    #
    #   dir_origin_path and dir_save_path are only active when mode='dir_predict'.
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   heatmap_save_path   Path to save the heatmap visualization.
    #                       Saved to model_data/ by default.
    #
    #   heatmap_save_path is only active when mode='heatmap'.
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    #-------------------------------------------------------------------------#
    #   simplify            Whether to use Simplify when exporting to ONNX.
    #   onnx_save_path      Path to save the exported ONNX model.
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode != "predict_onnx":
        yolo = YOLO()
    else:
        yolo = YOLO_ONNX()

    if mode == "predict":
        '''
        Tips for customizing single image prediction:
        1. To save the output image, use r_image.save("img.jpg") after detect_image().
        2. To get bounding box coordinates, go into yolo.detect_image() and read
           top, left, bottom, right in the drawing section.
        3. To crop detected objects, use the top/left/bottom/right values inside
           yolo.detect_image() to slice the original image array.
        4. To write extra text on the output image (e.g. object count), go into
           yolo.detect_image(), check predicted_class in the drawing section
           (e.g. if predicted_class == 'car':), count as needed, and use draw.text() to write.
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Could not read from camera or video. Check that the camera is connected or the video path is correct.")

        fps = 0.0
        while(True):
            t1 = time.time()
            # Read a frame
            ref, frame = capture.read()
            if not ref:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            frame = Image.fromarray(np.uint8(frame))
            # Run detection
            frame = np.array(yolo.detect_image(frame))
            # Convert RGB back to BGR for OpenCV display
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
        
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)
                
    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)

    elif mode == "predict_onnx":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")