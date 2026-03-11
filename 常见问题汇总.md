The blog post for this FAQ summary is at [https://blog.csdn.net/weixin_44791964/article/details/107517428](https://blog.csdn.net/weixin_44791964/article/details/107517428).

# FAQ Summary

## 1. Download Issues

### a. Code Download

**Q: Can you send me the code? Where can I download it?
A: The GitHub link is in the video description. Just copy it and you can go download it.**

**Q: Why does the downloaded code say the archive is corrupted?
A: Re-download it from GitHub.**

**Q: Why does the code I downloaded differ from what's shown in the video and blog?
A: I update the code frequently. The actual code in the repository is always the authoritative version.**

### b. Weights Download

**Q: Why are there no `.pth` or `.h5` files under `model_data` in the downloaded code?
A: I usually upload the weights to both GitHub and Baidu Netdisk. You can find the links in the README on GitHub.**

### c. Dataset Download

**Q: Where can I download the XXXX dataset?
A: Download links for datasets are generally listed in the README. If one is missing, feel free to contact me to add it — just open a GitHub issue.**

---

## 2. Environment Setup Issues

### a. GPU Series 20 and Below

**The PyTorch code requires PyTorch version 1.2.** Blog post: [https://blog.csdn.net/weixin_44791964/article/details/106037141](https://blog.csdn.net/weixin_44791964/article/details/106037141)

**The Keras code requires TensorFlow 1.13.2 and Keras 2.1.5.** Blog post: [https://blog.csdn.net/weixin_44791964/article/details/104702142](https://blog.csdn.net/weixin_44791964/article/details/104702142)

**The TF2 code requires TensorFlow 2.2.0; no separate Keras installation needed.** Blog post: [https://blog.csdn.net/weixin_44791964/article/details/109161493](https://blog.csdn.net/weixin_44791964/article/details/109161493)

**Q: Can I use version X of TensorFlow or PyTorch with your code?
A: It's best to use the versions I recommend — I provide setup tutorials too. I haven't tested other versions; they may cause issues, but generally only minor ones requiring small code adjustments.**

### b. GPU Series 30

Due to framework updates, the setup tutorials above do not apply to 30-series GPUs. The configurations I've tested and confirmed to work are:

**PyTorch code: PyTorch 1.7.0, CUDA 11.0, cuDNN 8.0.5.** Blog post: [https://blog.csdn.net/weixin_44791964/article/details/120668551](https://blog.csdn.net/weixin_44791964/article/details/120668551)

**Keras code: Cannot configure CUDA 11 on Windows 10. On Ubuntu, search online for instructions; use TensorFlow 1.15.4 and Keras 2.1.5 or 2.3.1 (note: minor API differences may require small code adjustments).**

**TF2 code: TensorFlow 2.4.0, CUDA 11.0, cuDNN 8.0.5.** Blog post: [https://blog.csdn.net/weixin_44791964/article/details/120657664](https://blog.csdn.net/weixin_44791964/article/details/120657664)

### c. CPU Environment

**PyTorch code: pytorch-cpu 1.2.** Blog post: [https://blog.csdn.net/weixin_44791964/article/details/120655098](https://blog.csdn.net/weixin_44791964/article/details/120655098)

**Keras code: tensorflow-cpu 1.13.2 and Keras 2.1.5.** Blog post: [https://blog.csdn.net/weixin_44791964/article/details/120653717](https://blog.csdn.net/weixin_44791964/article/details/120653717)

**TF2 code: tensorflow-cpu 2.2.0; no separate Keras needed.** Blog post: [https://blog.csdn.net/weixin_44791964/article/details/120656291](https://blog.csdn.net/weixin_44791964/article/details/120656291)

### d. GPU Utilization Issues

**Q: I installed tensorflow-gpu but training isn't using the GPU. Why?
A: Confirm tensorflow-gpu is installed by running `pip list`. Then check Task Manager or use the `nvidia-smi` command to see if the GPU is being used. In Task Manager, check VRAM usage.**

**Q: How can I tell if training is using the GPU?
A: Use the NVIDIA command-line tool. On Windows, open CMD and run `nvidia-smi` to view GPU utilization.**

If checking Task Manager, look at the GPU Performance section for VRAM usage — check the **Cuda** option, not Copy.

### e. "DLL load failed: The specified module could not be found"

**Q: I'm getting this error:**
```python
ImportError: DLL load failed: The specified module could not be found.
```
**A: If you haven't restarted, try that first. Otherwise, reinstall following the setup steps. If the issue persists, send me your GPU, CUDA, cuDNN, TF, and PyTorch versions via private message.**

### f. "No module" Errors (e.g., `no module named utils.utils`, `no module named 'matplotlib'`)

**Q: Why does it say `no module named utils.utils` (or `nets.yolo`, `nets.ssd`, etc.)?
A: `utils` does not need to be installed via pip — it's in the root directory of the repository I uploaded. This error means your working directory is wrong. Look up the concepts of relative and absolute paths.**

**Q: Why does it say `no module named matplotlib` (or `PIL`, `cv2`, etc.)?
A: That package isn't installed. Open a terminal and install it: `pip install matplotlib`**

**Q: I already installed opencv (or pillow, matplotlib) with pip, but I still get "no module named cv2". Why?
A: You installed it outside your active conda environment. Activate the correct conda environment first, then install.**

**Q: Why does it say `No module named 'torch'`?
A: Two possible reasons: (1) PyTorch genuinely isn't installed, or (2) it was installed in a different conda environment than the one currently active.**

**Q: Why does it say `No module named 'tensorflow'`?
A: Same as above.**

### g. CUDA Installation Failure

CUDA typically requires Visual Studio to be installed first. Visual Studio 2017 is sufficient.

### h. Ubuntu

**All code works on Ubuntu. I've tested on both Windows and Ubuntu.**

### i. VSCode Error Warnings

**Q: Why does VSCode show a bunch of errors?
A: I get them too, but they don't affect execution — it's a VSCode issue. If you want to avoid them, use PyCharm instead. Alternatively, set Python: Language Server to Pylance in VSCode settings.**

### j. Training and Prediction on CPU

**For Keras and TF2 code: simply install the CPU version of TensorFlow.**

**For PyTorch code: change `cuda=True` to `cuda=False`.**

### k. `tqdm` "no attribute 'pos'" Error

**Q: Getting `'tqdm' object has no attribute 'pos'`.
A: Reinstall tqdm with a different version.**

### l. `decode("utf-8")` Error

**Due to updates in the h5py library, installations may automatically pull in h5py 3.0.0+, which causes `decode("utf-8")` errors. After installing TensorFlow, be sure to downgrade h5py:**
```
pip install h5py==2.10.0
```

### m. `TypeError: __array__() takes 1 positional argument but 2 were given`

Fix by downgrading Pillow:
```
pip install pillow==8.2.0
```

### n. How to Check Your CUDA and cuDNN Versions

**Checking CUDA version on Windows:**
1. Open CMD.
2. Type `nvcc -V`.
3. The version appears in the line: `Cuda compilation tools, release XXXXXXXX`.

**Checking cuDNN version on Windows:**
1. Navigate to the CUDA installation directory and open the `include` folder.
2. Find `cudnn.h`.
3. Open it with a text editor and look for the `#define` lines:
```python
#define CUDNN_MAJOR 7
#define CUDNN_MINOR 4
#define CUDNN_PATCHLEVEL 1
```
This means cuDNN version 7.4.1.

### o. Setup Still Doesn't Work After Following Instructions

**Q: I followed your environment setup but it still doesn't work.
A: Please send me your GPU, CUDA, cuDNN, TF, and PyTorch versions via private message on Bilibili.**

### p. Other Errors

**Q: Getting `TypeError: cat() got an unexpected keyword argument 'axis'` or `AttributeError: 'Tensor' object has no attribute 'bool'`.
A: This is a version compatibility issue. Use torch 1.2 or above.**

**Many strange errors are version-related. It's strongly recommended to follow my video tutorials for setting up Keras and TensorFlow. For example, if you installed TensorFlow 2, don't ask why Keras-YOLO won't run — it obviously won't.**

---

## 3. Object Detection Library FAQ (Also Applies to Face Detection and Classification)

### a. Shape Mismatch Issues

#### 1) Shape mismatch during training

**Q: Why does running `train.py` give a shape mismatch error?
A: In Keras, because your number of classes differs from the original model's, the network structure changes slightly, causing minor shape mismatches at the tail end. This is expected.**

#### 2) Shape mismatch during prediction

**Q: Why does running `predict.py` give a shape mismatch error?**

##### i. In PyTorch:
`copying a param with shape torch.Size([75, 704, 1, 1]) from checkpoint`

##### ii. In Keras:
`Shapes are [1,1,1024,75] and [255,1024,1,1]. for 'Assign_360'...`

**A: The main causes are:
1. `classes_path` was not updated before training.
2. `model_path` was not updated.
3. `classes_path` was not updated for prediction.
Make sure your `model_path` and `classes_path` are consistent with each other, and also check `num_classes` or `classes_path` used during training!**

### b. Out of Memory (OOM / RuntimeError: CUDA out of memory)

**Q: The command line output flashes rapidly and shows OOM. Why?
A: You're running out of VRAM. Reduce `batch_size`. SSD uses the least VRAM — it's recommended for low-VRAM setups:**
- **2GB VRAM:** SSD, YOLOv4-Tiny
- **4GB VRAM:** YOLOv3
- **6GB VRAM:** YOLOv4, RetinaNet, M2Det, EfficientDet, Faster RCNN, etc.
- **8GB+ VRAM:** Any model

**Note: Due to BatchNorm2d, `batch_size` must be at least 2 (not 1).**

**Q: Getting `RuntimeError: CUDA out of memory. Tried to allocate 52.00 MiB (GPU 0; 15.90 GiB total capacity; 14.85 GiB already allocated; 51.88 MiB free; ...)`
A: You're out of VRAM in PyTorch. Same solution as above.**

**Q: Why did I run out of VRAM even though the GPU didn't seem to be used at all?
A: If it's OOM, the model never started training — that's why no GPU usage is shown.**

### c. Why Freeze/Unfreeze Training? Is It Necessary?

**Q: Why do we need frozen and unfrozen training phases?
A: It's optional. It's mainly to help users with limited hardware. If your machine can't handle full training, set `Freeze_Epoch` and `UnFreeze_Epoch` to the same value to only do frozen training.**

**This is also a transfer learning concept. The backbone's features are general-purpose, so freezing it speeds up training and prevents destroying pretrained weights.**

- **Frozen phase:** The backbone is frozen. VRAM usage is low; only fine-tuning the head.
- **Unfrozen phase:** The backbone is unfrozen. VRAM usage is higher; all parameters are updated.

### d. My Loss Is Very High / Very Low — Is That a Problem?

**Q: My network isn't converging. My loss is XXXX.
A: Loss varies by network. Loss is only an indicator of convergence, not a measure of model quality. My YOLO code doesn't normalize loss, so the values appear high. What matters is whether the loss is decreasing and whether predictions look reasonable.**

### e. Why Does My Trained Model Produce No Predictions?

**Q: My training results are poor — no bounding boxes (or inaccurate ones). Why?**

Consider the following:
1. **Target info:** Check if `2007_train.txt` contains target annotations. If not, fix `voc_annotation.py`.
2. **Dataset size:** If fewer than 500 samples, consider collecting more data. Test different models to confirm the dataset is valid.
3. **Unfreezing:** If your dataset distribution differs significantly from standard images, try unfreezing training to adjust the backbone.
4. **Network choice:** Some networks like SSD aren't suitable for small objects due to fixed anchor sizes.
5. **Training duration:** Don't judge after just a few epochs — train to completion with default parameters.
6. **Follow the steps:** Check that you followed all the setup steps (e.g., did you update `classes` in `voc_annotation.py`?).
7. **Loss:** Loss is just a convergence indicator, not a quality metric. What matters is whether it's decreasing.
8. **Backbone modification:** If you changed the backbone without pretrained weights, convergence will be difficult.

### f. Why Is My Calculated mAP 0?

**Q: My training results are poor — mAP is 0. Why?**

Similar to the above. Check:
1. First try running `predict.py`. If predictions look fine, the issue is likely `classes_path` in `get_map.py`. If predictions also fail, use the same checklist as in section (e).

### g. GBK Encoding Error (`'gbk' codec can't decode byte`)

**Q: Getting this error:**
```python
UnicodeDecodeError: 'gbk' codec can't decode byte 0xa6 in position 446: illegal multibyte sequence
```
**A: Don't use Chinese characters in labels or paths. If you must, change the file encoding to `utf-8` when opening files.**

### h. My Images Have a Resolution of xxx×xxx — Will It Work?

**Q: My images are xxx×xxx resolution. Can I use them?
A: Yes. The code automatically performs resize and data augmentation.**

### i. I Want to Do Data Augmentation — How?

**Q: How do I do data augmentation?
A: The code already includes automatic resize and data augmentation.**

### j. Multi-GPU Training

**Q: How do I train on multiple GPUs?
A: Most PyTorch code can use multi-GPU directly. For Keras, search online — it's not complicated. I don't have multiple GPUs to test with, so you'll need to figure that part out yourselves.**

### k. Can I Train on Grayscale Images?

**Q: Can I train (or predict) on grayscale images?
A: Most of my libraries convert grayscale to RGB before training and prediction. If you encounter issues, try converting the result of `Image.open()` to RGB inside `get_random_data`, and do the same during prediction. (For reference only.)**

### l. Resuming Training from a Checkpoint

**Q: I've already trained for a few epochs. Can I continue from where I left off?
A: Yes. Load the saved weights just like you would load pretrained weights. Trained weights are usually saved in the `logs` folder. Set `model_path` to the path of the checkpoint you want to resume from.**

### m. Using Pretrained Weights for a Different Dataset

**Q: If I'm training on a different dataset, can I still use the pretrained weights?
A: Yes. Pretrained weights are transferable across datasets because features are universal. In 99% of cases, pretrained weights are essential. Without them, weights are too random and feature extraction suffers, leading to poor results.**

### n. Training from Scratch

**Q: How do I train without pretrained weights?
A: Check the comments. In most code, set `model_path = ''` and `Freeze_Train = False`. If `model_path` has no effect, comment out the pretrained weight loading code.**

### o. Why Does Training from Scratch Give Such Poor Results?

**Q: Why is training without pretrained weights so bad?
A: Randomly initialized weights are poor for feature extraction, which leads to bad model performance. Pretrained weights are very important.**

**Q: I modified the network — can I still use the pretrained weights?
A: If you changed the backbone and it's not a standard network, pretrained weights generally won't work. You'd need to either manually match weights by shape, or pretrain from scratch. If you only changed the latter part of the network, the backbone pretrained weights are still usable. For PyTorch, modify the weight loading logic to match by shape. For Keras, use `by_name=True, skip_mismatch=True`.**

Example weight-matching code for PyTorch:
```python
# Speed up model training
print('Loading weights into state dict...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dict = model.state_dict()
pretrained_dict = torch.load(model_path, map_location=device)
a = {}
for k, v in pretrained_dict.items():
    try:    
        if np.shape(model_dict[k]) == np.shape(v):
            a[k] = v
    except:
        pass
model_dict.update(a)
model.load_state_dict(model_dict)
print('Finished!')
```

**Q: Why is training from scratch (or with a modified backbone) so bad?
A: Training from scratch generally gives very poor results due to random initialization. It is strongly, strongly, strongly discouraged! If you must, first train a classification model on ImageNet to obtain backbone weights, then use those as a starting point.**

**Q: How do I train from scratch on the model?
A: Without sufficient compute and tuning experience, training from scratch is meaningless. Random initialization gives very poor feature extraction. If you insist on training from scratch:**
- Do not load pretrained weights.
- Do not use frozen training — comment out the freeze code.

### p. Where Do Your Pretrained Weights Come From?

**Q: If you can't train from scratch, where do your weights come from?
A: Some weights are converted from official sources; others I trained myself. The ImageNet backbone weights I use are all from official sources.**

### q. Video and Camera Detection

**Q: How do I use webcam detection?
A: Modify the parameters in `predict.py` for webcam detection. There's also a video explaining the approach in detail.**

**Q: How do I do video detection?
A: Same as above.**

### r. Saving Detection Results

**Q: How do I save detected images?
A: Object detection generally uses PIL's `Image` class. Look up how to save with PIL. Check the comments in `predict.py` for details.**

**Q: How do I save detection results from video?
A: Check the comments in `predict.py`.**

### s. Batch Processing a Folder of Images

**Q: How do I process all images in a folder?
A: Use `os.listdir` to list all images, then follow the detection logic in `predict.py`. Check its comments for details.**

**Q: How do I batch process and save results?
A: Use `os.listdir` to list images, detect using the logic in `predict.py`, and save using PIL's `Image.save()` (or cv2's image-saving function if the library uses cv2). Check `predict.py` comments for details.**

### t. Path Errors (`No such file or directory`, `Permission denied`)

**Q: Getting errors like:**
```python
FileNotFoundError: [Errno 2] No such file or directory
StopIteration: [Errno 13] Permission denied: 'D:\\...\\VOCdevkit/VOC2007'
```
**A: Check your folder paths to confirm the files exist. Also check `2007_train.txt` to verify the file paths are correct.**

Key points about paths:
- **No spaces in folder names.**
- **Be careful about relative vs. absolute paths.**
- **Search online to learn more about path handling.**

**Almost all path issues come down to the root directory. Make sure you understand relative paths!**

### u. Comparison with the Original Implementation

**Q: The original code does XXX, why does yours do XXX?
A: Well... that's kind of the point — I'm not the original author...**

**Q: How does your code compare to the original? Can it match the original performance?
A: It basically can. I've tested it on the VOC dataset. I don't have a powerful GPU to test on COCO.**

**Q: Have you implemented all the tricks from YOLOv4? How much does it differ from the original?
A: Not all improvements are implemented. YOLOv4 uses so many tricks it's hard to replicate everything. I implemented the ones I found most effective. Even the original author's code doesn't use SAM (the attention module). As for the comparison with the original, I can't train on COCO, but users who have tried report the gap is small.**

### v. Detection Speed

**Q: What FPS can this achieve? Can it reach XX FPS?
A: FPS depends on your hardware. Better hardware = faster speed.**

**Q: Is my detection speed of xxx normal? Can it be improved?
A: Depends on your hardware. For speed improvements without hardware upgrades, you'd need to modify the network.**

**Q: Why does my server only show ~10 FPS for YOLOv4?
A: Check whether tensorflow-gpu or PyTorch GPU is correctly installed. If it is, use `time.time()` to profile `detect_image` and find which part is the bottleneck (not just the network — drawing bounding boxes and other post-processing also takes time).**

**Q: Why can't I reach the FPS claimed in the paper?
A: Verify GPU installation. Also note that some papers use batch prediction, which I haven't implemented.**

### w. Predicted Image Not Displayed

**Q: Why doesn't the code display the image after prediction? It only prints results in the terminal.
A: Install an image viewer on your system.**

### x. Evaluation Metrics (mAP, PR Curve, Recall, Precision)

**Q: How do I calculate mAP?
A: Follow the mAP measurement video — the workflow is the same.**

**Q: What is `MINOVERLAP` in `get_map.py`? Is it IoU?
A: Yes, it's the IoU threshold. If the overlap between a predicted box and the ground truth box exceeds `MINOVERLAP`, the prediction is counted as correct.**

**Q: Why is `self.confidence` (or `self.score`) in `get_map.py` set so low?
A: Watch the mAP video for the theory — all predictions need to be collected before drawing the PR curve.**

**Q: How do I plot PR curves?
A: Watch the mAP video. The output already includes PR curves.**

**Q: How do I calculate Recall and Precision?
A: These metrics are relative to a specific confidence threshold and are also computed during mAP calculation.**

### y. COCO Dataset Training

**Q: How do I train on the COCO dataset?
A: The format for COCO training annotation files is the same as in qqwweee's YOLOv3 repo — refer to that.**

### z. Model Optimization / Improving Performance

**Q: How do I modify the model to write a small paper?
A: I recommend studying the differences between YOLOv3 and YOLOv4, then reading the YOLOv4 paper — it's a great reference for tricks and tuning. My general advice: study classic models, identify their key structural innovations, and incorporate them.**

### aa. Focal Loss

**Q: Do you have code that uses Focal Loss with YOLO? Does it help?
A: Many people have tried it — the improvement is minimal (sometimes even worse). YOLO already has its own positive/negative sample balancing mechanism.** For modifying the code, read through it carefully yourself.

### ab. Deployment (ONNX, TensorRT, etc.)

I haven't deployed to mobile or embedded devices, so I'm not familiar with most deployment questions...

---

## 4. Semantic Segmentation Library FAQ

### a. Shape Mismatch Issues

#### 1) Shape mismatch during training

**Q: Why does `train.py` give a shape mismatch error?
A: In Keras, since your number of classes differs from the original model, the network structure changes slightly, causing minor shape mismatches at the output.**

#### 2) Shape mismatch during prediction

**Q: Why does `predict.py` give a shape mismatch error?**

##### i. In PyTorch:
`copying a param with shape torch.Size([75, 704, 1, 1]) from checkpoint`

##### ii. In Keras:
`Shapes are [1,1,1024,75] and [255,1024,1,1]. for 'Assign_360'...`

**A: Main causes:
1. `num_classes` in `train.py` was not updated.
2. `num_classes` for prediction was not updated.
3. `model_path` for prediction was not updated.
Check all of these carefully — `num_classes` must be consistent for both training and prediction!**

### b. Out of Memory (OOM / RuntimeError: CUDA out of memory)

**Q: The command line flashes rapidly and shows OOM. Why?
A: You're out of VRAM in Keras. Reduce `batch_size`.**

**Note: Due to BatchNorm2d, `batch_size` must be at least 2.**

**Q: Getting `RuntimeError: CUDA out of memory...`?
A: Same as above, but in PyTorch.**

**Q: Why did I run out of VRAM when GPU usage wasn't even shown?
A: The model never started — the OOM happened before training began.**

### c. Freeze/Unfreeze Training

**Q: Why freeze and unfreeze training?
A: Same reasoning as in the object detection section. It's optional but helps with limited hardware. The frozen phase fine-tunes the head; the unfrozen phase trains the whole network.**

### d. Loss Is Very High / Very Low

**Q: My network isn't converging. My loss is XXXX.
A: Same as in the object detection section — loss varies by network. What matters is whether it's decreasing, not its absolute value.**

### e. Trained Model Produces No Predictions

**Q: My training results are poor — no predictions (or inaccurate ones).**

Consider:
1. **Dataset quality:** Most important. Under 500 samples? Add more. Check that label pixel values match their class index. A common mistake is white-on-black labels where the target pixel value is 255 — it must be 1.
2. **Unfreezing:** If your data is very different from standard images, try unfreezing training.
3. **Network choice:** Try different networks.
4. **Training duration:** Train to completion with default settings before judging.
5. **Follow the steps:** Double-check you followed everything correctly.
6. **Loss:** Only matters that it's converging, not its absolute value.

**Q: My predictions for small targets are inaccurate.**
**A: For DeepLab and PSPNet, try changing `downsample_factor` from 16 to 8 to reduce excessive downsampling.**

### f. Why Is My Calculated mIoU 0?

**Q: My mIoU is 0.**

Similar to (e). Check:
1. Dataset quality and label correctness (pixel values must match class indices).
2. Whether unfreeze training is needed.
3. Network choice.
4. Training duration.
5. Whether you followed all steps.
6. Loss convergence.

### g. GBK Encoding Error

**Q: Getting a GBK encoding error:**
```python
UnicodeDecodeError: 'gbk' codec can't decode byte 0xa6...
```
**A: Avoid Chinese characters in labels and paths. If necessary, open files with `encoding='utf-8'`.**

### h. Image Resolution

**Q: My images are xxx×xxx — can I use them?
A: Yes. The code automatically resizes and augments images.**

### i. Data Augmentation

**Q: How do I do data augmentation?
A: It's already built in. The code automatically performs resize and data augmentation.**

### j. Multi-GPU Training

**Q: How do I train on multiple GPUs?
A: Same as in the object detection section.**

### k. Grayscale Images

**Q: Can I train/predict on grayscale images?
A: Same as in the object detection section — try converting to RGB inside `get_random_data`.**

### l. Resuming Training

**Q: I've trained for a few epochs already — can I continue from that checkpoint?
A: Yes. Load the saved weights from the `logs` folder by setting `model_path` to the checkpoint path.**

### m. Pretrained Weights for a Different Dataset

**Q: Can I use the pretrained weights for a different dataset?
A: Yes. Pretrained features are universal across datasets. In 99% of cases, pretrained weights are essential.**

### n. Training from Scratch

**Q: How do I train without pretrained weights?
A: Set `model_path = ''` and `Freeze_Train = False`. Comment out the weight-loading code if needed.**

### o. Poor Results from Scratch / Modified Backbone

**Q: Why are results so bad without pretrained weights?
A: Random initialization leads to poor feature extraction. Pretrained weights are critical.**

**Q: I modified the backbone — can I still use pretrained weights?
A: If you changed the backbone to a non-standard architecture, the weights likely won't transfer. Use shape-based matching or pretrain separately. If you only changed the head, the backbone weights still apply.**

**Q: Why is training from scratch so bad? / I modified the backbone and results are poor.**
**A: Random weights give terrible feature extraction. It is very strongly recommended NOT to train from scratch. If you must, first train a classification model on ImageNet, then use the backbone weights to bootstrap your model.**

**If you insist on training from scratch:**
- Do not load pretrained weights.
- Do not freeze any layers.

### p. Where Do Your Weights Come From?

**Q: If training from scratch doesn't work, where did you get your weights?
A: Some are converted from official sources; some I trained myself. The ImageNet backbone weights are all from official sources.**

### q. Video and Camera Detection

**Q: How do I use webcam detection?
A: Same as in the object detection section.**

### r. Saving Detection Results

**Q: How do I save detected images / video results?
A: Check the comments in `predict.py`.**

### s. Batch Processing a Folder

**Q: How do I process all images in a folder (with or without saving)?
A: Use `os.listdir` to list files. Follow the detection logic in `predict.py`. For saving, use PIL's `Image.save()` or cv2's image-saving function.**

### t. Path Errors

**Q: Getting `No such file or directory` or `Permission denied` errors?
A: Check all folder paths. No spaces in folder names. Understand relative vs. absolute paths.**

### u. Comparison with Original

**Same as in the object detection section.**

### v. Detection Speed

**Same as in the object detection section.**

### w. Predicted Image Not Displayed

**Q: Why doesn't the predicted image show up?
A: Install an image viewer on your system.**

### x. mIoU Evaluation

**Q: How do I calculate mIoU?
A: Refer to the mIoU measurement section in the video.**

**Q: How do I calculate Recall and Precision?
A: These require understanding the confusion matrix. Current code doesn't directly compute them — you'll need to implement it yourself.**

### y. Model Optimization

**Q: How do I modify the model to improve performance?
A: Study the YOLOv4 paper — it's an excellent reference for tricks. My general advice: read classic models, understand their key innovations, and incorporate them.**

### z. Deployment (ONNX, TensorRT, etc.)

I haven't deployed to mobile/embedded devices, so I'm not familiar with most deployment questions...

---

## 5. Community / Chat Group

**Q: Do you have a QQ group or Discord?
A: No — I don't have time to manage a chat group.**

---

## 6. How to Learn Deep Learning

**Q: What's your learning path? I'm a beginner — where do I start?
A: A few things to note:**
1. I'm not an expert — there's a lot I don't know, and my path may not work for everyone.
2. My lab doesn't focus on deep learning, so most of what I know is self-taught. I can't guarantee it's all correct.
3. In my experience, self-study is key.

My personal learning path: I started with Morvan's Python tutorials, then got into TensorFlow, Keras, and PyTorch. After getting comfortable with those, I studied SSD and YOLO, then explored many classic CNN architectures, and eventually started learning from diverse codebases. My method is reading code line by line — understanding execution flow, how feature map shapes change, and so on. It takes a lot of time, but there's no shortcut.