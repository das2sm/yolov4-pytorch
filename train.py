#-------------------------------------#
#   Train the dataset
#-------------------------------------#
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (YOLOLoss, get_lr_scheduler, set_optimizer_lr,
                                weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (get_anchors, get_classes, seed_everything,
                         show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch

'''
Important notes for training your own object detection model:

1. Before training, carefully verify that your data format meets requirements.
   This repo expects VOC format: input images and XML label files.
   - Images must be .jpg. Size does not need to be fixed — they are auto-resized before training.
   - Grayscale images are automatically converted to RGB. No manual changes needed.
   - If your images are not .jpg, batch-convert them before training.
   - Labels must be .xml files containing bounding box and class info, one per image.

2. Loss value is used to judge convergence. What matters is the trend — validation loss
   should decrease over time. If it stops changing, the model has converged.
   The absolute value of loss doesn't matter much; it depends on how the loss is calculated
   and does not need to be close to 0. If you want smaller-looking loss values, divide
   by 10000 inside the loss function.
   Loss values are saved in logs/loss_%Y_%m_%d_%H_%M_%S/ during training.

3. Trained weight files are saved in the logs/ folder. Each epoch contains multiple steps;
   each step performs one gradient descent update.
   Weights are NOT saved if you only ran a few steps. Make sure you understand the
   difference between Epoch and Step.
'''
if __name__ == "__main__":
    #---------------------------------#
    #   Cuda    Whether to use CUDA.
    #           Set to False if no GPU is available.
    #---------------------------------#
    Cuda = True
    #----------------------------------------------#
    #   Seed    Random seed for reproducibility.
    #           Ensures identical results across independent training runs.
    #----------------------------------------------#
    seed = 11
    #---------------------------------------------------------------------#
    #   distributed     Whether to use single-machine multi-GPU distributed training.
    #                   Terminal commands only supported on Ubuntu.
    #                   Use CUDA_VISIBLE_DEVICES on Ubuntu to specify GPUs.
    #                   On Windows, DP mode is used by default (all GPUs). DDP is not supported.
    #
    #   DP mode:
    #       Set             distributed = False
    #       Run:            CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode:
    #       Set             distributed = True
    #       Run:            CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     Whether to use SyncBatchNorm. Only useful in DDP multi-GPU mode.
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training.
    #               Reduces VRAM usage by ~half. Requires PyTorch >= 1.7.1.
    #---------------------------------------------------------------------#
    fp16            = False
    #---------------------------------------------------------------------#
    #   classes_path    Path to the txt file in model_data listing your classes.
    #                   Must be updated to match your dataset before training.
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/coco_classes.txt'
    #---------------------------------------------------------------------#
    #   anchors_path    Path to the anchor box txt file. Generally not modified.
    #   anchors_mask    Helps the code locate the correct anchor boxes. Generally not modified.
    #---------------------------------------------------------------------#
    anchors_path    = 'model_data/yolo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    #----------------------------------------------------------------------------------------------------------------------------#
    #   See README for weight file download links.
    #   Pretrained weights are universal across datasets because features are general-purpose.
    #   The most important part of pretrained weights is the backbone feature extractor.
    #   Pretrained weights are essential in 99% of cases. Without them, backbone weights are
    #   too random, feature extraction is ineffective, and training results will be poor.
    #
    #   If training is interrupted, set model_path to a checkpoint in logs/ to resume.
    #   Also update the Freeze/Unfreeze epoch parameters to maintain continuity.
    #
    #   When model_path = '', the full model weights are not loaded.
    #
    #   This loads the full model weights in train.py; the pretrained flag below does not affect this.
    #   To train from backbone pretrained weights only: set model_path = '', pretrained = True.
    #   To train from scratch: set model_path = '', pretrained = False, Freeze_Train = False.
    #
    #   Training from scratch is strongly, strongly, strongly discouraged — random weights
    #   lead to poor feature extraction and bad results.
    #   If you must train from scratch, two options:
    #   1. Use Mosaic augmentation with UnFreeze_Epoch >= 300, batch >= 16, and a large dataset (10k+ images).
    #      Set mosaic=True and train with random initialization. Still worse than using pretrained weights.
    #      (Suitable for large datasets like COCO.)
    #   2. First train a classification model on ImageNet to get backbone weights,
    #      then use those backbone weights as the starting point.
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/yolo4_voc_weights.pth'
    #------------------------------------------------------#
    #   input_shape     Input image size. Must be a multiple of 32.
    #------------------------------------------------------#
    input_shape     = [416, 416]
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      Whether to use pretrained backbone weights.
    #                   Backbone weights are loaded during model construction.
    #                   If model_path is set, backbone weights are already included — pretrained has no effect.
    #                   If model_path = '' and pretrained = True, only the backbone is loaded.
    #                   If model_path = '', pretrained = False, Freeze_Train = False: train fully from scratch.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = False
    #------------------------------------------------------------------#
    #   mosaic              Mosaic data augmentation.
    #   mosaic_prob         Probability of applying mosaic per step. Default: 50%.
    #
    #   mixup               Whether to use mixup augmentation. Only active when mosaic=True.
    #                       Applied only to mosaic-augmented images.
    #   mixup_prob          Probability of applying mixup after mosaic. Default: 50%.
    #                       Total mixup probability = mosaic_prob * mixup_prob.
    #
    #   special_aug_ratio   Inspired by YoloX. Mosaic-generated images deviate significantly
    #                       from the natural image distribution.
    #                       When mosaic=True, mosaic is only applied within this ratio of epochs.
    #                       Default: first 70% of epochs (e.g., 70 out of 100 epochs).
    #
    #   Cosine annealing parameters are set in lr_decay_type below.
    #------------------------------------------------------------------#
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    #------------------------------------------------------------------#
    #   label_smoothing     Label smoothing. Typically <= 0.01 (e.g., 0.01 or 0.005).
    #------------------------------------------------------------------#
    label_smoothing     = 0

    #----------------------------------------------------------------------------------------------------------------------------#
    #   Training has two phases: frozen and unfrozen.
    #   The frozen phase is designed for users with limited GPU resources.
    #   Frozen training uses less VRAM. If your GPU is very weak, set Freeze_Epoch = UnFreeze_Epoch
    #   to only perform frozen training.
    #
    #   Recommended parameter settings:
    #
    #   (1) Starting from full model pretrained weights:
    #       Adam:
    #           Init_Epoch=0, Freeze_Epoch=50, UnFreeze_Epoch=100, Freeze_Train=True,  optimizer_type='adam', Init_lr=1e-3, weight_decay=0  (frozen)
    #           Init_Epoch=0,                  UnFreeze_Epoch=100, Freeze_Train=False, optimizer_type='adam', Init_lr=1e-3, weight_decay=0  (no freeze)
    #       SGD:
    #           Init_Epoch=0, Freeze_Epoch=50, UnFreeze_Epoch=300, Freeze_Train=True,  optimizer_type='sgd',  Init_lr=1e-2, weight_decay=5e-4  (frozen)
    #           Init_Epoch=0,                  UnFreeze_Epoch=300, Freeze_Train=False, optimizer_type='sgd',  Init_lr=1e-2, weight_decay=5e-4  (no freeze)
    #       UnFreeze_Epoch can be adjusted between 100–300.
    #
    #   (2) Starting from backbone pretrained weights only:
    #       Adam:
    #           Init_Epoch=0, Freeze_Epoch=50, UnFreeze_Epoch=100, Freeze_Train=True,  optimizer_type='adam', Init_lr=1e-3, weight_decay=0  (frozen)
    #           Init_Epoch=0,                  UnFreeze_Epoch=100, Freeze_Train=False, optimizer_type='adam', Init_lr=1e-3, weight_decay=0  (no freeze)
    #       SGD:
    #           Init_Epoch=0, Freeze_Epoch=50, UnFreeze_Epoch=300, Freeze_Train=True,  optimizer_type='sgd',  Init_lr=1e-2, weight_decay=5e-4  (frozen)
    #           Init_Epoch=0,                  UnFreeze_Epoch=300, Freeze_Train=False, optimizer_type='sgd',  Init_lr=1e-2, weight_decay=5e-4  (no freeze)
    #       Note: Backbone weights may not be optimal for detection, so more epochs are needed.
    #             UnFreeze_Epoch: 150–300. Both YOLOv5 and YOLOX recommend 300.
    #             Adam converges faster than SGD, so UnFreeze_Epoch can theoretically be smaller,
    #             but more epochs are still recommended.
    #
    #   (3) Training from scratch:
    #       Init_Epoch=0, UnFreeze_Epoch>=300, Unfreeze_batch_size>=16, Freeze_Train=False
    #       Keep UnFreeze_Epoch >= 300. Use optimizer_type='sgd', Init_lr=1e-2, mosaic=True.
    #
    #   (4) batch_size guidelines:
    #       Larger is better within your GPU's VRAM limits.
    #       OOM errors are not related to dataset size — reduce batch_size if you get OOM.
    #       Due to BatchNorm, batch_size must be at least 2 (cannot be 1).
    #       Freeze_batch_size is typically 1–2x Unfreeze_batch_size. Large differences
    #       are not recommended as they affect automatic learning rate adjustment.
    #----------------------------------------------------------------------------------------------------------------------------#

    #------------------------------------------------------------------#
    #   Frozen phase training parameters.
    #   The backbone is frozen — the feature extractor does not change.
    #   Lower VRAM usage; only the head is fine-tuned.
    #
    #   Init_Epoch          Starting epoch. Can be set higher than Freeze_Epoch, e.g.:
    #                       Init_Epoch=60, Freeze_Epoch=50, UnFreeze_Epoch=100
    #                       This skips the frozen phase and starts directly at epoch 60
    #                       with the corresponding learning rate. (Used for resuming training.)
    #   Freeze_Epoch        Number of epochs to train with frozen backbone.
    #                       (Ignored when Freeze_Train=False)
    #   Freeze_batch_size   Batch size during frozen training.
    #                       (Ignored when Freeze_Train=False)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 2  # Was 50 - changed to 2 for sanity check
    Freeze_batch_size   = 8
    #------------------------------------------------------------------#
    #   Unfrozen phase training parameters.
    #   The backbone is unfrozen — the feature extractor will update.
    #   Higher VRAM usage; all parameters are trained.
    #
    #   UnFreeze_Epoch          Total number of training epochs.
    #                           SGD needs more epochs to converge; use a larger value.
    #                           Adam can use a relatively smaller UnFreeze_Epoch.
    #   Unfreeze_batch_size     Batch size after unfreezing.
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 4  # Was 300 - changed to 4 for sanity check
    Unfreeze_batch_size = 4
    #------------------------------------------------------------------#
    #   Freeze_Train    Whether to do frozen training first.
    #                   Default: freeze backbone first, then unfreeze.
    #------------------------------------------------------------------#
    Freeze_Train        = True
    
    #------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, LR schedule
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         Maximum learning rate.
    #   Min_lr          Minimum learning rate. Default: 1% of Init_lr.
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  Optimizer to use: 'adam' or 'sgd'.
    #                   Recommended Init_lr=1e-3 for Adam.
    #                   Recommended Init_lr=1e-2 for SGD.
    #   momentum        Momentum parameter for the optimizer.
    #   weight_decay    Weight decay to prevent overfitting.
    #                   Adam can cause incorrect weight decay — set to 0 when using Adam.
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   Learning rate decay schedule: 'step' or 'cos' (cosine annealing).
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   focal_loss      Whether to use Focal Loss to balance positive/negative samples.
    #   focal_alpha     Focal Loss positive/negative sample balance parameter.
    #   focal_gamma     Focal Loss easy/hard sample balance parameter.
    #------------------------------------------------------------------#
    focal_loss          = False
    focal_alpha         = 0.25
    focal_gamma         = 2
    #------------------------------------------------------------------#
    #   iou_type        IoU loss type to use: 'ciou' or 'siou'.
    #------------------------------------------------------------------#
    iou_type            = 'ciou'
    #------------------------------------------------------------------#
    #   save_period     Save weights every N epochs.
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #   save_dir        Directory to save weights and log files.
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       Whether to evaluate on the validation set during training.
    #                   Install pycocotools for a better evaluation experience.
    #   eval_period     Evaluate every N epochs. Frequent evaluation slows training
    #                   significantly — avoid setting this too low.
    #   Note: mAP computed here differs from get_map.py for two reasons:
    #   (1) This evaluates on the validation set, not test set.
    #   (2) Evaluation settings here are conservative to keep it fast.
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 10
    #------------------------------------------------------------------#
    #   num_workers     Number of worker threads for data loading.
    #                   More workers = faster data loading, but more RAM usage.
    #                   Set to 2 or 0 if RAM is limited.
    #------------------------------------------------------------------#
    num_workers         = 4

    #------------------------------------------------------#
    #   train_annotation_path   Path to training image paths and labels.
    #   val_annotation_path     Path to validation image paths and labels.
    #------------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    seed_everything(seed)
    #------------------------------------------------------#
    #   Set up GPU devices
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0
        
    #------------------------------------------------------#
    #   Load classes and anchors
    #------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)
    
    #------------------------------------------------------#
    #   Build the YOLO model
    #------------------------------------------------------#
    model = YoloBody(anchors_mask, num_classes, pretrained = pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
        #   See README for weight file download links.
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   Load weights by matching keys and shapes between
        #   the pretrained weights and the current model.
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   Display keys that failed to load
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mNote: It is normal for head weights to fail loading. Backbone weights failing to load indicates an error.\033[0m")

    #----------------------#
    #   Define loss function
    #----------------------#
    yolo_loss    = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing, focal_loss, focal_alpha, focal_gamma, iou_type)
    #----------------------#
    #   Set up loss logging
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    #------------------------------------------------------------------#
    #   torch 1.2 does not support AMP. Use torch >= 1.7.1 for fp16.
    #   In torch 1.2, this will show "could not be resolved".
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   Sync BatchNorm across GPUs (multi-GPU)
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   Multi-GPU parallel run
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #---------------------------#
    #   Read annotation txt files
    #---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path = classes_path, anchors_path = anchors_path, anchors_mask = anchors_mask, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        #---------------------------------------------------------#
        #   Total epochs = number of full passes through the dataset.
        #   Total steps  = total number of gradient descent updates.
        #   Each epoch contains multiple steps; each step does one gradient update.
        #   The minimum recommended epochs are calculated based on the unfrozen phase only.
        #----------------------------------------------------------#
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('Dataset is too small to train. Please add more data.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] When using %s optimizer, total training steps should be at least %d.\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] Current dataset: %d samples, Unfreeze_batch_size=%d, UnFreeze_Epoch=%d → total steps=%d.\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Total steps (%d) is below the recommended minimum (%d). Consider setting UnFreeze_Epoch to %d.\033[0m"%(total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    #   Backbone features are general-purpose. Frozen training speeds up early training
    #   and prevents backbone weights from being corrupted in the initial phase.
    #   Init_Epoch    = starting epoch
    #   Freeze_Epoch  = epoch at which frozen phase ends
    #   UnFreeze_Epoch = total epochs
    #   If you get OOM or CUDA out of memory, reduce batch_size.
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   Freeze part of the model for frozen training phase
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        #-------------------------------------------------------------------#
        #   If not freezing, set batch_size directly to Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   Adaptively adjust learning rate based on current batch_size
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type in ['adam', 'adamw'] else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type in ['adam', 'adamw'] else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   Build optimizer based on optimizer_type
        #---------------------------------------#
        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'adamw' : optim.AdamW(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        #---------------------------------------#
        #   Get the learning rate decay function
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   Calculate number of steps per epoch
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset is too small to continue training. Please add more data.")

        #---------------------------------------#
        #   Build dataset loaders
        #---------------------------------------#
        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, \
                                        mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        #----------------------#
        #   Track mAP curve during evaluation
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   Start model training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   If the model has a frozen section,
            #   unfreeze it when the time comes and update settings
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   Adaptively adjust learning rate based on new batch_size
                #-------------------------------------------------------------------#
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type in ['adam', 'adamw'] else 5e-2
                lr_limit_min    = 3e-4 if optimizer_type in ['adam', 'adamw'] else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   Update learning rate decay function
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset is too small to continue training. Please add more data.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node
                    
                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            gen.dataset.epoch_now       = epoch
            gen_val.dataset.epoch_now   = epoch

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
                        
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()