import cv2
import numpy as np
import time
import threading
import queue
from rknnlite.api import RKNNLite

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
RKNN_MODEL      = 'models.rknn'
CLASSES_PATH    = 'cls_classes.txt'
INPUT_SIZE      = 416
CONF_THRESH     = 0.25
NMS_THRESH      = 0.3
CAMERA_ID       = '/dev/video1'
NUM_CORES       = 3
QUEUE_SIZE      = 6
MAX_FPS_SAMPLES = 30

ANCHORS = [
    [12, 16],  [19, 36],  [40, 28],
    [36, 75],  [76, 55],  [72, 146],
    [142, 110],[192, 243],[459, 401]
]
ANCHOR_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

# -------------------------------------------------------
# Load class names
# -------------------------------------------------------
with open(CLASSES_PATH, 'r') as f:
    CLASSES = [line.strip() for line in f.readlines() if line.strip()]
NUM_CLASSES = len(CLASSES)
print(f'Loaded {NUM_CLASSES} classes: {CLASSES}')

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)


# -------------------------------------------------------
# Threaded camera capture
# -------------------------------------------------------
class CameraThread:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            raise RuntimeError(f'Failed to open camera: {src}')

        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        print('Camera thread started')

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def release(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


# -------------------------------------------------------
# Preprocessing
# -------------------------------------------------------
def preprocess(image, input_size):
    h, w = image.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.full((input_size, input_size, 3), 128, dtype=np.uint8)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return padded, scale, pad_x, pad_y


# -------------------------------------------------------
# Sigmoid
# -------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -88, 88)))


# -------------------------------------------------------
# Vectorized decode
# -------------------------------------------------------
def decode_output_vectorized(output, anchors, input_size, num_classes, conf_thresh):
    output = output[0].astype(np.float32)
    num_anchors = len(anchors)
    grid_h, grid_w = output.shape[1], output.shape[2]

    output = output.reshape(num_anchors, 5 + num_classes, grid_h, grid_w)
    output = output.transpose(0, 2, 3, 1)

    grid_x = np.arange(grid_w).reshape(1, 1, grid_w)
    grid_y = np.arange(grid_h).reshape(1, grid_h, 1)
    anchors_arr = np.array(anchors, dtype=np.float32)

    tx = sigmoid(output[..., 0])
    ty = sigmoid(output[..., 1])
    tw = output[..., 2]
    th = output[..., 3]
    obj_conf = sigmoid(output[..., 4])
    cls_probs = sigmoid(output[..., 5:])

    cx = (tx + grid_x) / grid_w
    cy = (ty + grid_y) / grid_h
    bw = anchors_arr[:, 0].reshape(num_anchors, 1, 1) * np.exp(np.clip(tw, -88, 88)) / input_size
    bh = anchors_arr[:, 1].reshape(num_anchors, 1, 1) * np.exp(np.clip(th, -88, 88)) / input_size

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    cls_ids = np.argmax(cls_probs, axis=-1)
    cls_scores = np.max(cls_probs, axis=-1)
    scores = obj_conf * cls_scores

    mask = scores >= conf_thresh
    if not np.any(mask):
        return [], [], []

    boxes = np.stack([x1, y1, x2, y2], axis=-1)[mask]
    final_scores = scores[mask]
    final_cls_ids = cls_ids[mask]

    return boxes.tolist(), final_scores.tolist(), final_cls_ids.tolist()


# -------------------------------------------------------
# NMS
# -------------------------------------------------------
def nms(boxes, scores, class_ids, nms_thresh):
    if len(boxes) == 0:
        return [], [], []

    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)

    keep_boxes = []
    keep_scores = []
    keep_class_ids = []

    for cls_id in np.unique(class_ids):
        mask = class_ids == cls_id
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]

        indices = cv2.dnn.NMSBoxes(
            cls_boxes.tolist(),
            cls_scores.tolist(),
            score_threshold=0.0,
            nms_threshold=nms_thresh
        )

        if len(indices) > 0:
            for idx in indices.flatten():
                keep_boxes.append(cls_boxes[idx])
                keep_scores.append(cls_scores[idx])
                keep_class_ids.append(cls_id)

    return keep_boxes, keep_scores, keep_class_ids


# -------------------------------------------------------
# Draw detections
# -------------------------------------------------------
def draw_detections(image, boxes, scores, class_ids, scale, pad_x, pad_y, input_size):
    h, w = image.shape[:2]

    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box

        x1 = (x1 * input_size - pad_x) / scale
        y1 = (y1 * input_size - pad_y) / scale
        x2 = (x2 * input_size - pad_x) / scale
        y2 = (y2 * input_size - pad_y) / scale

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        color = tuple(int(c) for c in COLORS[cls_id])
        label = f'{CLASSES[cls_id]}: {score:.2f}'

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(image, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


# -------------------------------------------------------
# NPU worker — one per core
# -------------------------------------------------------
class NPUWorker:
    def __init__(self, worker_id, core_mask, input_queue, output_queue):
        self.worker_id    = worker_id
        self.input_queue  = input_queue
        self.output_queue = output_queue
        self.running      = True

        # Per-worker FPS tracking
        self._last_fps_time = time.time()
        self._frame_count   = 0

        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(RKNN_MODEL)
        if ret != 0:
            raise RuntimeError(f'Worker {worker_id}: Failed to load RKNN model')
        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f'Worker {worker_id}: Failed to init NPU core')
        print(f'Worker {worker_id} initialized on NPU core {worker_id}')

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                item = self.input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            frame_id, frame, input_img, scale, pad_x, pad_y = item

            outputs = self.rknn.inference(inputs=[input_img])

            all_boxes, all_scores, all_class_ids = [], [], []
            for i, output in enumerate(outputs):
                anchor_indices = ANCHOR_MASK[i]
                anchors = [ANCHORS[j] for j in anchor_indices]
                boxes, scores, class_ids = decode_output_vectorized(
                    output, anchors, INPUT_SIZE, NUM_CLASSES, CONF_THRESH
                )
                all_boxes.extend(boxes)
                all_scores.extend(scores)
                all_class_ids.extend(class_ids)

            final_boxes, final_scores, final_class_ids = nms(
                all_boxes, all_scores, all_class_ids, NMS_THRESH
            )

            # Track per-worker throughput
            self._frame_count += 1
            elapsed = time.time() - self._last_fps_time
            if elapsed >= 1.0:
                print(f'Worker {self.worker_id} throughput: {self._frame_count / elapsed:.1f} FPS')
                self._frame_count = 0
                self._last_fps_time = time.time()

            self.output_queue.put((frame_id, frame, final_boxes, final_scores,
                                   final_class_ids, scale, pad_x, pad_y))
            self.input_queue.task_done()

    def release(self):
        self.running = False
        self.thread.join(timeout=3)
        self.rknn.release()


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    input_queue  = queue.Queue(maxsize=QUEUE_SIZE)
    output_queue = queue.Queue(maxsize=QUEUE_SIZE)

    # Start 3 NPU workers
    core_masks = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
    workers = []
    for i, core in enumerate(core_masks):
        workers.append(NPUWorker(i, core, input_queue, output_queue))

    cam = CameraThread(CAMERA_ID)
    print('All workers ready. Press Q to quit.')

    # Smooth FPS tracking — measures actual detection throughput
    fps_history    = []
    last_result    = None
    frame_id       = 0
    last_result_time = time.time()

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        # Preprocess and push to input queue
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img, scale, pad_x, pad_y = preprocess(img_rgb, INPUT_SIZE)
        input_img = np.expand_dims(input_img, axis=0)

        try:
            input_queue.put_nowait((frame_id, frame, input_img, scale, pad_x, pad_y))
            frame_id += 1
        except queue.Full:
            pass  # drop frame if all cores are busy

        # Get latest detection result (non-blocking)
        try:
            result = output_queue.get_nowait()
            # Measure FPS based on how fast results come out
            now = time.time()
            fps_history.append(1.0 / max(now - last_result_time, 1e-6))
            if len(fps_history) > MAX_FPS_SAMPLES:
                fps_history.pop(0)
            last_result_time = now
            last_result = result
        except queue.Empty:
            pass

        # Display
        display_frame = frame.copy()
        if last_result is not None:
            _, _, final_boxes, final_scores, final_class_ids, r_scale, r_pad_x, r_pad_y = last_result
            display_frame = draw_detections(
                display_frame, final_boxes, final_scores, final_class_ids,
                r_scale, r_pad_x, r_pad_y, INPUT_SIZE
            )

        # Smooth FPS = average of last 30 detection results
        smooth_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0
        cv2.putText(display_frame, f'FPS: {smooth_fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f'Queue: {input_queue.qsize()}/{QUEUE_SIZE}', (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('YOLOv4 Traffic Signs - 3 Core NPU', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Shutting down...')
    cam.release()
    for w in workers:
        w.release()
    cv2.destroyAllWindows()
    print('Done')


if __name__ == '__main__':
    main()
