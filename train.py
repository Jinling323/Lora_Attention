from utils.regression_trainer_cosine_multibatch import RegTrainer
import argparse
import os
import torch
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--model-name', default='vgg19_trans', help='the name of the model')
    parser.add_argument('--data-dir', default='/media/mmslab5090/SSD2/crowd counting test/sha/clean',
                        help='clean pretraining root containing train/ and val/')
    parser.add_argument('--train-dir', default='/media/mmslab5090/SSD2/crowd counting test/sha/hazy',
                        help='LoRA training root containing train/; empty value falls back to hazy/ beside data-dir')
    parser.add_argument('--val-dir', default='/media/mmslab5090/SSD2/crowd counting test/sha/mix',
                        help='LoRA validation root containing val/; empty value falls back to mix/ beside data-dir')
    parser.add_argument('--save-dir', default='model',
                        help='directory to save models.')
    parser.add_argument('--save-all', type=bool, default=False,
                        help='whether to save all best model')
    parser.add_argument('--lr', type=float, default=5*1e-6,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='the weight decay')
    parser.add_argument('--resume', default='',
                        help='.pth: baseline for new LoRA fine-tuning; .tar: continue training')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=1200,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=600,
                        help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')

    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=16,
                        help='downsample ratio')

    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=0.15,
                        help='background ratio')
    parser.add_argument('--lora', action='store_true', help='fine-tune only LoRA adapters')
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=float, default=8.0)
    parser.add_argument('--lora-attention', choices=['all', 'qv'], default='all')
    parser.add_argument('--num-layers', type=int, default=4, help='encoder blocks; checkpoint depth must match')
    args = parser.parse_args()
    if args.resume and os.path.splitext(args.resume)[1].lower() not in ('.pth', '.tar'):
        parser.error('--resume must be a .pth or .tar file')
    if args.lora and not args.resume:
        parser.error('--lora requires --resume with a complete checkpoint')
    if args.lora_rank <= 0 or args.num_layers <= 0:
        parser.error('rank and num-layers must be positive')
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
