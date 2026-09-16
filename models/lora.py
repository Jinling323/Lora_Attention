"""Single-adapter LoRA for the crowd Transformer (no routing or experts)."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LowRankUpdate(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('LoRA rank and alpha must be positive and finite')
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank
        nn.init.normal_(self.A)

    def forward(self, x):
        return F.linear(F.linear(x, self.A), self.B) * self.scale


class LoRALinear(nn.Linear):
    @classmethod
    def from_linear(cls, linear, rank, alpha):
        result = cls(linear.in_features, linear.out_features,
                     bias=linear.bias is not None)
        result.weight = linear.weight
        result.bias = linear.bias
        result.lora = LowRankUpdate(linear.in_features, linear.out_features, rank, alpha)
        result.to(device=linear.weight.device, dtype=linear.weight.dtype)
        result.train(linear.training)
        return result

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + self.lora(x)


def inject_lora(model, rank=8, alpha=8.0, attention='all'):
    if attention not in ('all', 'qv'):
        raise ValueError('attention must be all or qv')
    for layer in model.encoder.layers:
        attn = layer.self_attn
        if hasattr(attn, 'lora_in'):
            raise ValueError('LoRA has already been installed')
        indices = range(9) if attention == 'all' else (0, 2)
        attn.lora_in = nn.ModuleDict({
            str(i): LowRankUpdate(attn.embed_dim, attn.embed_dim, rank, alpha)
            for i in indices
        }).to(device=attn.in_proj_weight.device, dtype=attn.in_proj_weight.dtype)
        if attention == 'all':
            attn.out_proj = LoRALinear.from_linear(attn.out_proj, rank, alpha)
        layer.linear1 = LoRALinear.from_linear(layer.linear1, rank, alpha)
        layer.linear2 = LoRALinear.from_linear(layer.linear2, rank, alpha)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = '.lora.' in name or '.lora_in.' in name


def checkpoint_parts(checkpoint):
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict'], checkpoint.get('model_config')
    return checkpoint, None


def build_model(state=None, config=None, num_layers=4, lora=None):
    """Load base weights strictly before injection; restore adapters with metadata."""
    from models.vgg_c_multibatch import vgg19_trans
    if state is not None:
        layer_ids = {int(k.split('.')[2]) for k in state if k.startswith('encoder.layers.')}
        if not layer_ids:
            raise ValueError('Expected a complete crowd-counting checkpoint, including encoder')
        num_layers = max(layer_ids) + 1
    if config is not None:
        if state is not None and config['num_layers'] != num_layers:
            raise ValueError('Checkpoint layer count does not match its configuration')
        num_layers = config['num_layers']
        saved_lora = config.get('lora')
    else:
        saved_lora = None
    if state is not None and any('.lora' in k for k in state) and saved_lora is None:
        raise ValueError('LoRA checkpoint is missing model_config metadata')
    if lora and state is None:
        raise ValueError('LoRA requires a complete pretrained crowd-counting checkpoint')
    model = vgg19_trans(num_layers=num_layers, pretrained=state is None)
    if saved_lora:
        if lora is not None and lora != saved_lora:
            raise ValueError('Requested LoRA settings differ from checkpoint')
        inject_lora(model, **saved_lora)
        model.load_state_dict(state, strict=True)
        lora = saved_lora
    else:
        if state is not None:
            model.load_state_dict(state, strict=True)
        if lora:
            if state is None:
                raise ValueError('LoRA requires a complete pretrained crowd-counting checkpoint')
            inject_lora(model, **lora)
    return model, {'num_layers': num_layers, 'lora': lora}


def build_training_model(args):
    """A .pth starts fresh LoRA training; a .tar restores the saved run."""
    from pathlib import Path
    state = config = checkpoint = None
    suffix = Path(args.resume).suffix.lower() if args.resume else ''
    if suffix not in ('', '.pth', '.tar'):
        raise ValueError('--resume must be a .pth or .tar file')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state, config = checkpoint_parts(checkpoint)
        layer_ids = {int(k.split('.')[2]) for k in state if k.startswith('encoder.layers.')}
        if not layer_ids or max(layer_ids) + 1 != args.num_layers:
            raise ValueError('Checkpoint encoder depth must match --num-layers; '
                             'use a matching baseline (default: 4 blocks)')
    lora = None
    if suffix == '.pth':
        if (config and config.get('lora')) or any('.lora' in k for k in state):
            raise ValueError('.pth initialization requires baseline weights without LoRA; '
                             'use the training .tar to resume LoRA')
        lora = dict(rank=args.lora_rank, alpha=args.lora_alpha,
                    attention=args.lora_attention)
    elif suffix == '.tar':
        required = ('model_state_dict', 'optimizer_state_dict', 'epoch')
        if not all(key in checkpoint for key in required):
            raise ValueError('.tar must contain model, optimizer and epoch state')
        if args.lora and not (config and config.get('lora')):
            raise ValueError('This .tar is a baseline run; use a baseline .pth to start LoRA')
    elif args.lora:
        raise ValueError('LoRA requires a baseline .pth or a LoRA training .tar')
    model, model_config = build_model(state, config, num_layers=args.num_layers, lora=lora)
    return model, model_config, checkpoint if suffix == '.tar' else None
