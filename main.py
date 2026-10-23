import torch.optim as optim


from inference import inference_ds_monai
from train import train_single_epoch, train_single_epoch_monai
from utils.model_utils import get_model, get_standard_model
from utils.dataset_utils import get_dataset
from segment_anything.utils.transforms import ResizeLongestSide

from utils.utils import str2bool, set_seed, disable_batchnorm_running_stats
import wandb
from utils.scheduler import WarmupCosineSchedule, WarmupLinearSchedule



import os
import torch
import argparse
import sys
import shlex
from segment_anything import sam_model_registry
from utils.utils import str2bool
from utils.saver import Saver

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
print('\n'+'='*60)
print('='*60)



def main(args=None, sam_args=None, saver=None):
    print('\n'+'='*60)
    print("[MAIN] Device configuration")
    if str(args['device']).lower() == 'cuda':
        if not torch.cuda.is_available():
            print(f"[WARNING] CUDA requested but not available. Falling back to CPU.")
            args['device'] = torch.device("cpu")
        else:
            num_gpus = torch.cuda.device_count()
            device_id = int(args.get('device_id', 0))
            if device_id >= num_gpus or device_id < 0:
                print(f"[WARNING] device_id={device_id} is out of range (only {num_gpus} GPU(s) available). Using device 0.")
                device_id = 0
                args['device_id'] = 0    
            args['device'] = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(device_id) 

            gpu_name = torch.cuda.get_device_name(device_id)
            gpu_props = torch.cuda.get_device_properties(device_id)
            total_vram_gb = gpu_props.total_memory / (1024**3)

            print(f"[MAIN] Device        : {args['device']}")
            print(f"[MAIN] GPU name      : {gpu_name}")
            print(f"[MAIN] Total VRAM    : {total_vram_gb:.2f} GB")
            print(f"[MAIN] CUDA version  : {torch.version.cuda}")
            print(f"[MAIN] PyTorch version: {torch.__version__}")

            cc = (gpu_props.major, gpu_props.minor)
            print(f"[MAIN] Compute capability: {cc[0]}.{cc[1]}")

            if cc < (6, 0): print(f"[WARNING] GPU compute capability {cc[0]}.{cc[1]} is below 6.0. Mixed precision (AMP) may not work properly.")
            if cc < (7, 0) and args.get('use_cuda_amp', False): print(f"[WARNING] GPU compute capability {cc[0]}.{cc[1]} is below 7.0. Tensor Cores not available; AMP will be limited.")

            try:
                free_vram, total_vram = torch.cuda.mem_get_info(device_id)
                free_vram_gb = free_vram / (1024**3)
                print(f"[MAIN] Free VRAM     : {free_vram_gb:.2f} GB")
            
                if args.get('use_sam', False) and free_vram_gb < 8:
                    print(f"[WARNING] Free VRAM ({free_vram_gb:.2f} GB) may be insufficient for Phase 1 with SAM (ViT-B). Consider:")
                    print(f"         - Smaller batch size (-bs)")
                    print(f"         - Using --use_cuda_amp True")
                    print(f"         - Using a smaller SAM variant (sam_vit_b)")
            except Exception: pass 
    else:
        args['device'] = torch.device("cpu")
        print(f"[MAIN] Device        : CPU")
        print(f"[MAIN] PyTorch version: {torch.__version__}")
        if args.get('use_sam', False):
            print(f"[WARNING] Running Phase 1 with SAM on CPU will be extremely slow!")
            print(f"         Consider using a GPU for training.")
        if args.get('use_cuda_amp', False):
            print(f"[WARNING] AMP is enabled but device is CPU. Disabling AMP.")
            args['use_cuda_amp'] = False
    print('='*60)
        
    print('\n'+'='*60)
    if args['use_sam']:
        print("[MAIN] SAM model (Phase 1: SAM-guided prompt learning) Configuration")
        try: sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
        except Exception as e: raise RuntimeError(f"Failed to load SAM '{sam_args['model_type']}' from '{sam_args['sam_checkpoint']}': {e}")
        sam.to(device=args['device'])
        sam.eval()
        for param in sam.parameters(): param.requires_grad = False
        sam_params = sum(p.numel() for p in sam.parameters())
        sam_memory_mb = sum(p.numel() * p.element_size() for p in sam.parameters()) / (1024**2)
        img_dim = sam.image_encoder.img_size
        print(f"[MAIN] SAM input resolution: {img_dim}x{img_dim}")
        print(f"[MAIN] SAM Parameters: {sam_params:,}")
        print(f"[MAIN] SAM Memory: {sam_memory_mb:.2f} MB")
        print(f"[MAIN] SAM Device: {args['device']}")
        print(f"[MAIN] SAM Mode: Frozen (eval)")
        if args['disable_batchnorm_running_stats']:
            disable_batchnorm_running_stats(sam)
            print("[MAIN] BatchNorm running stats disabled for SAM.")
    else:
        sam = None
        img_dim = int(args['Idim'])
        print("[MAIN] SAM is disabled. Using standard segmentor directly.")
        print(f"[MAIN] Using Idim = {img_dim}×{img_dim} for prompt learner input")
    print('='*60)   
    input('stop') 
    
    ds, ds_val = get_dataset(args, transform, img_dim)
    img_ch = ds.dataset[0][0][args['image_key']].shape[0]
    
    if not args['use_standard_net']:
        model = get_model(args, sam, img_ch)
    else:
        model = get_standard_model(args, img_ch)
        
    if args['disable_batchnorm_running_stats']:
            disable_batchnorm_running_stats(model)
    
    if not args['use_sam']:
        if not args['segmentor_finetune_backbone']:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.segmentor.parameters():
                param.requires_grad = True
            params_to_optimize = model.segmentor.parameters()
        else:
            for param in model.parameters():
                param.requires_grad = True
            params_to_optimize = model.parameters()
    else:
        # If using SAM, optimize all model parameters
        params_to_optimize = model.parameters()

    if args['optim']=='Adam':
        optimizer = optim.Adam(params_to_optimize,
                            lr=float(args['learning_rate']),
                            weight_decay=float(args['WD']))
    elif args['optim']=='SGD':
        optimizer = optim.SGD(params_to_optimize,
                            lr=float(args['learning_rate']),
                            weight_decay=float(args['WD']),
                            momentum=float(args['momentum']))
    elif args['optim']=='AdamW':
        optimizer = optim.AdamW(params_to_optimize,
                            lr=float(args['learning_rate']),
                            weight_decay=float(args['WD']))
    if args['use_scheduler']:
        steps_per_epoch = len(ds)
        if args['scheduler']=='StepLR':
            step_size = steps_per_epoch * args['scheduler_step']
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=args['scheduler_gamma'])
        elif args['scheduler']== 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=args['scheduler_gamma'])
        elif args['scheduler']== 'ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args['scheduler_gamma'])
        elif args['scheduler']== 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args['scheduler_gamma'], patience=args['scheduler_patience'], verbose=True)
        elif args['scheduler']== 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['schedCos_Tmax'], eta_min=args['schedCos_eta_min'], last_epoch=args['schedCos_last_epoch'])            
        elif args['scheduler']== 'WarmupCosineSchedule':
            warmup_epochs = args['warmup_epochs']
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=steps_per_epoch * warmup_epochs, t_total=steps_per_epoch * args['epoches'], cycles=args['warmup_cycles'], last_epoch=args['schedWarmupCos_last_epoch'])
            print('Using WarmupCosineSchedule with {} warmup epochs, {} cycles, {} steps per epoch'.format(warmup_epochs, args['warmup_cycles'], steps_per_epoch))
        elif args['scheduler']== 'WarmupLinearSchedule':
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=args['epoches'])
    else:
        scheduler = None

    
    if args['wandb_watch']:
        wandb.watch(model, log='all', log_freq=args['wandb_watch_freq'], log_graph=True)
    
    best = 0
    path_best = os.path.join(saver.path,'best.csv')
    f_best = open(path_best, 'w')

    scaler = torch.cuda.amp.GradScaler(enabled=args['use_cuda_amp'])
    
    for epoch in range(int(args['epoches'])):
        if args['task'] in ['pancreas', 'spleen', 'prostate', 'brats', 'mslesseg']:
            avg_loss, avg_dice, avg_iou= train_single_epoch_monai(ds, model.train(), sam.eval() if args['use_sam'] else None, optimizer, transform, epoch, args, saver, scheduler, scaler)
        else:
            avg_loss= train_single_epoch(ds, model.train(), sam.eval(), optimizer, transform, epoch, args)
        saver.log_loss('train_loss', avg_loss, epoch)
        saver.log_loss('train_dice', avg_dice, epoch)
        saver.log_loss('train_iou', avg_iou, epoch)
        with torch.no_grad():
            if args['task'] in ['pancreas', 'spleen', 'prostate', 'brats', 'mslesseg']:
                dice_true, dice_false, IoU_val, val_loss = inference_ds_monai(ds_val, model.eval(), sam, transform, epoch, args, saver)
                saver.log_loss('val_loss', val_loss, epoch)
                saver.log_loss('IoU', IoU_val, epoch)
                saver.log_loss('Dice/include_true', dice_true, epoch)
                saver.log_loss('Dice/include_false', dice_false, epoch)
                
            else:
                dice, IoU_val = inference_ds(ds_val, model.eval(), sam, transform, epoch, args)
                saver.log_loss('IoU', IoU_val, epoch)
                saver.log_loss('Dice', dice, epoch)
            if args['best_metric']=='dice_f':
                metric = dice_false
            elif args['best_metric']=='dice_t':
                metric = dice_true
            elif args['best_metric']=='IoU':
                metric = IoU_val
            else:
                raise ValueError('Best metric not recognized')
            if metric > best:
                torch.save(model.state_dict(), args['path_best'])
                best = metric
                print('best results: ' + str(best))
                f_best.write(str(epoch) + ',' + str(best) + '\n')
                f_best.flush()
        
        if epoch % int(args['save_every']) == 0 or epoch == int(args['epoches'])-1:
            saver.save_model(model, 'net_last', epoch)
        
    f_best.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
============================================================
            POLYP SEGMENTATION FRAMEWORK
============================================================
Method : SAM-Guided Prompt Learning for Polyp Segmentation
Task   : Automated Polyp Detection and Segmentation
Input  : Colonoscopy RGB Images (2D)
Output : Binary Polyp Segmentation Masks
============================================================

This framework leverages Segment Anything Model (SAM) during 
training to guide a dense prompt learner. At inference, SAM 
is replaced by a lightweight aggregator for efficient deployment.

For usage examples, see README.md
============================================================\n""", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--strict_determinism', type=str2bool, default=False, help='Enable strict determinism (slower but reproducible). Use True for final thesis experiments.')
    parser.add_argument('--exp_name', required=True, help='Experiment name (required)')
    parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'], help='WandB mode')
    parser.add_argument('--wandb_project', default='PS_PROJECT', help='WandB project name')
    parser.add_argument('--wandb_entity', default=None, help='WandB entity (username or team)')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='Wandb tags')
    parser.add_argument('--use_sam', type=str2bool, default=True, help='Use SAM (phase 1) vs aggregator only (phase 2)')
    parser.add_argument('--sam_ckpt', type=str, default='sam_vit_b', choices = ['sam_vit_b', 'medsam_vit_b', 'sam_vit_h'], help='SAM checkpoint')
    parser.add_argument('--sam_version', type=str, default='vit_b', choices = ['vit_b', 'vit_h'], help='SAM version')
    parser.add_argument('--device_id', type=int, default=0, help='GPU index')
    parser.add_argument('--device', default='cuda', help="Device: 'cuda' or 'cpu'")
    parser.add_argument('-Idim', '--Idim', type=int, default=512, help='Input image size (resize resolution)')
    args = vars(parser.parse_args())
    print(parser.description)
    
    """ parser.add_argument('-lr', '--learning_rate', default=0.01, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=2, help='batch_size', required=False)
    parser.add_argument('-epoches', '--epoches', type=int, default=5000, help='number of epoches', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-WD', '--WD', default=1e-4, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='mslesseg', help='Task type (e.g., pancreas, spleen, prostate, brats, mslesseg, tooth)', required=False)
    parser.add_argument('-depth_wise', '--depth_wise', type = str2bool, default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, type=int, help='image size', required=False)
    parser.add_argument('-Idim', '--Idim', default=512, help='image size', required=False)
    parser.add_argument('-rotate', '--rotate', default=22, help='image size', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='image size', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='image size', required=False)

    parser.add_argument('--data_dir', type=str, default='', help='Path to the dataset directory', required=False)
    parser.add_argument('--split_path', type=str, default='3DAutoSam_Pancreas/Dataset.json', help='Path to the split file')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=10, help='Save model every n epochs')
    parser.add_argument('--image_key', type=str, default='image',help='Modality to use for training')
    parser.add_argument('--mask_key', type=str, default='mask', help='Mask modality')
    parser.add_argument('--spatial_crop_size', type=int, default=224, help='Spatial crop size')
    parser.add_argument('--pos_rand_crop', type=float, default=0.75, help='Positive ratio for random crop')
    parser.add_argument('--neg_rand_crop', type=float, default=0.25, help='Negative ratio for random crop')
    parser.add_argument('--axcodes', type=str, default='RAS', help='Axial codes')
    parser.add_argument('--num_slices', type=int, default=1, help='Number of slices')
    parser.add_argument('--n_fold', type=int, default=0, help='Specify fold')
    parser.add_argument('--criterion', type=str, default='dice', choices=['dice', 'dice_ce'], help='Criterion')
    parser.add_argument('--include_background', type=str2bool, default=False, help='Include background in loss')
    parser.add_argument('--acc_steps', type=int, default=2, help='Number of accumulation steps')
    parser.add_argument('--cache_rate', type=float, default=.7, help='Cache rate')
    parser.add_argument('--optim', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum (if SGD is used)')
    parser.add_argument('--use_scheduler', type=str2bool, default=False, help='If use scheduler or not')
    parser.add_argument('--scheduler', type=str, default='StepLR', choices=['StepLR', 'MultiStepLR', 'ExponentialLR', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'WarmupCosineSchedule', 'WarmupLinearSchedule'], help='Scheduler')
    parser.add_argument('--scheduler_step', type=int, default=100, help='Scheduler step for StepLR, and first step of MultiStepLR')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Scheduler gamma')
    parser.add_argument('--scheduler_patience', type=int, default=10, help='ReduceLROnPlateau Scheduler patience')
    parser.add_argument('--schedCos_Tmax', type=int, default=10, help='CosineAnnealingLR Tmax')
    parser.add_argument('--schedCos_eta_min', type=float, default=0.0001, help='CosineAnnealingLR eta_min')
    parser.add_argument('--schedCos_last_epoch', type=int, default=-1, help='CosineAnnealingLR last_epoch')
    parser.add_argument('--best_metric', type=str, default='IoU', help='Best metric') #until now, only IoU is implemented as best metric to save the best model
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--warmup_cycles', type=float, default=0.5, help='Warmup cycles')
    parser.add_argument('--schedWarmupCos_last_epoch', type=int, default=-1, help='WarmupCosineSchedule last_epoch (use only if you are resuming a training)')
    parser.add_argument('--use_cuda_amp', type=str2bool, default=True, help='Use mixed precision training')

    parser.add_argument('--save_train_images', type=str2bool, default=True, help='Save images')
    parser.add_argument('--save_val_images', type=str2bool, default=True, help='Save images')
    parser.add_argument('--theashold_discretize', type=float, default=0.01, help='Threshold for discretization')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Checkpoint path to continue training')
    parser.add_argument('--pretrained', type=str2bool, default=True, help='Pretrained model')
    parser.add_argument('--model_emb', type=str, default='HardNet', help='Model embedding')

    parser.add_argument('--remove_maxpool', type=str2bool, default=False, help='Remove first maxpool when ResNet is used as embedding model')
    
    parser.add_argument('--use_standard_net', type=str2bool, default=False, help='Use standard net')
    parser.add_argument('--net', type=str, default='HardNetSegmentor', help='Net')
    parser.add_argument('--net_segmentor', type=str, default='PointwiseConv', help='Net segmentor')
    parser.add_argument('--sam_zeroshot', type=str2bool, default=False, help='SAM zero shot')
    parser.add_argument('--upsample_factor_list', type=int, nargs='+', default=[2, 2], help='Upsample factor list for ConvUpsample segmentor')
    

    parser.add_argument('--wandb_watch', type=str2bool, default=False, help='Wandb watch')
    parser.add_argument('--wandb_watch_freq', type=int, default=10, help='Wandb watch frequency')

    parser.add_argument('--out_channels', type=int, default=2, help='Output channels')
    parser.add_argument('--label_values', type=int, nargs='+', default=[1, 2, 4], help='Label values to discretize mask (primarily for the BRATS dataset; may not apply to other datasets)')
    parser.add_argument('--center_crop', type=str2bool, default=False, help='Center crop the input images')
    parser.add_argument('--train_with_crop', type=str2bool, default=True, help='Train with crop')

    parser.add_argument('--segmentor_finetune_backbone', type=str2bool, default=False, help='If we want to train the miniSegmentor finetuning also the backbone')
    parser.add_argument('--decoder_channels', type=int, nargs='+', default=[], help='Decoder channels for UnetLike segmentor/aggregator')
    
    
    parser.add_argument('--disable_batchnorm_running_stats', type=str2bool, default=False, help='Disable batchnorm running stats')
    
    parser.add_argument('--seed', type=int, default=42, help='Seed')"""
    

    set_seed(args['seed'], strict=args['strict_determinism'])

    args['experiment_name'] = args['exp_name']
    output_root = os.path.join(PROJECT_DIR, 'results')

    saver = Saver(output_folder=output_root, args=args)
    cmd = f"{sys.executable} " + " ".join(shlex.quote(arg) for arg in sys.argv)
    saver.log_cmd(cmd)

    args['exp_folder'] = saver.path
    args['last_ckpt_path'] = os.path.join(saver.ckpt_path, 'net_last.pth')
    args['best_ckpt_path'] = os.path.join(saver.ckpt_path, 'net_best.pth')
    args['vis_folder'] = saver.vis_path
    saver.log_hparams(args)

    print('\n'+'='*60)
    if args['use_sam']:
        sam_ckpt = os.path.join(PROJECT_DIR, 'cp', f"{args['sam_ckpt']}.pth")
        if not os.path.exists(sam_ckpt): raise FileNotFoundError(f"SAM checkpoint not found: {sam_ckpt}\n Please download {args['sam_ckpt']}.pth into the cp/ folder.")
        
        sam_args = {
            'sam_checkpoint': sam_ckpt,
            'model_type': args['sam_version'],
            'generator_args': {
                'points_per_side': 8,
                'pred_iou_thresh': 0.95,
                'stability_score_thresh': 0.7,
                'crop_n_layers': 0,
                'crop_n_points_downscale_factor': 2,
                'min_mask_region_area': 0,
                'point_grids': None,
                'box_nms_thresh': 0.7,
            },
            'gpu_id': args['device_id'],
        }
        print(f"[MAIN] SAM checkpoint path: {sam_ckpt}")
        print(f"[MAIN] SAM Model type : {args['sam_version']}")
    else:
        sam_args = None
        print("[MAIN] SAM is disabled.")
    print('='*60)

    main(args=args, sam_args=sam_args, saver=saver)
