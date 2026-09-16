# Boosting-Crowd-Counting-via-Multifaceted-Attention
Official Implement of CVPR 2022 paper 'Boosting Crowd Counting via Multifaceted Attention'

[arxiv](https://arxiv.org/pdf/2203.02636.pdf) | [知乎](https://zhuanlan.zhihu.com/p/478023612) | [B站](https://www.bilibili.com/video/BV13Y411u7r5?share_source=copy_web)

![image](structure.png)

## Train
1. Dowload Dataset JHU++ or UCF-QNRF.
2. Preprocess them by 'preprocess_dataset.py' or 'preprocess_dataset_ucf.py'.
3. Change the path to where your data and models are located in 'Train.py'.
4. Run 'Train.py'
5. Wait patiently for the program to finish.
6. Then you will get a good counting model!


## Test
1. Dowload Dataset JHU++ or UCF-QNRF.
2. Preprocess them by 'preprocess_dataset.py' or 'preprocess_dataset_ucf.py'.
3. JHU Model [Link](https://drive.google.com/file/d/14piGsWRFy9BSXI1Jv9zRxypDxpOHbwCY/view?usp=sharing); UCF Model [Link](https://drive.google.com/file/d/1Y2WU0kIlZq3x28JZskvGx1cuZt0KQkXF/view?usp=sharing)
4. Change the path to where your data and models are located in 'Test.py'.
5. Run 'Test.py'.


## Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{lin2022boosting,
  title={Boosting Crowd Counting via Multifaceted Attention},
  author={Lin, Hui and Ma, Zhiheng and Ji, Rongrong and Wang, Yaowei and Hong, Xiaopeng},
  booktitle={CVPR},
  year={2022}
}
```


## Single-adapter LoRA

`train.py` uses the multibatch model with 4 encoder blocks by default.
Each block contains attention and an FFN with two Linear layers.
Training checkpoints must match `--num-layers` (default 4); layers are never silently discarded. The DataLoader format is unchanged;
validation still uses batch size 1.

Start LoRA from a **complete trained crowd-counting checkpoint**, not ImageNet
VGG weights alone:

```bash
python train.py --data-dir /path/to/dataset --lora \
  --resume /path/to/base.pth --batch-size 2 \
  --lora-rank 8 --lora-alpha 8 --lora-attention all
```

Each encoder layer gets independent low-rank updates for all nine global/local
input projections, the output projection, and both FFN linear layers. `qv`
selects only global Q/V in attention, while retaining both FFN adapters.
There is one adapter per target, with no MoE or router. Original weights,
biases, LayerNorm, VGG and the regression head are frozen. Only LoRA A/B train.
A is Gaussian-initialized, B starts at zero, and scaling is alpha/r.
Applying LoRA to the local projections and FFN extends the original paper's
main Q/V experiments; it is not a claimed reproduction of those experiments.

Without `--resume` or `--lora`, training starts a full-parameter baseline.
A `.pth` passed to `--resume` automatically enables fresh LoRA fine-tuning,
even without `--lora`, with a new optimizer and epoch 0. If no
complete checkpoint is available, train a baseline first. Learning rate remains
configurable with `--lr`; tune it for your dataset.

```bash
python train.py --data-dir /path/to/dataset --resume /path/to/epoch_ckpt.tar
python test.py --data-dir /path/to/dataset --save-dir /path/to/best_model.pth
```

New best-model and epoch checkpoints include full model weights and
`model_config` (depth and LoRA settings); epoch checkpoints also include optimizer
and epoch state. Resume and test restore the saved adapter configuration without
repeating LoRA flags. Older plain baseline state dictionaries are supported.
Use `--resume base.pth` to initialize new adapters from baseline weights.
Use `--resume epoch_ckpt.tar` to restore model, optimizer and epoch; saved LoRA
settings are used automatically. A LoRA best-model `.pth` is for evaluation,
not baseline initialization; continue its run using the corresponding `.tar`. Legacy visualization scripts still expect plain baseline weights;
use `test.py` for LoRA evaluation.

CPU verification: `python -m unittest discover -s tests -v`.


## Automatic clean / hazy / mix datasets

Set the `--data-dir` default in `train.py` once to the **clean root**.
Pretraining reads `data-dir/train` and `data-dir/val` directly. LoRA training
reads the sibling `hazy/train` and `mix/val` folders automatically:

```text
/data/
  clean/       <-- data-dir
    train/
    val/
  hazy/
    train/
  mix/
    val/
```

`--train-dir` and `--val-dir` can override the sibling roots if stored elsewhere;
their defaults can also be edited once in `train.py`. Each image folder contains
`.jpg` images and matching preprocessed `.npy` annotations.

```bash
# Clean pretraining: uses the data-dir default
python train.py
# Hazy LoRA training, mix validation: automatic with baseline .pth
python train.py --resume /path/to/clean/best_model.pth
# Continue LoRA: dataset choice follows the model configuration in the .tar
python train.py --resume /path/to/lora/20_ckpt.tar
```

A baseline `.tar` continues on clean; a LoRA `.tar` continues on hazy/mix.
Resolved paths and image counts are logged. Model depth remains 4 by default.
The existing `--val-start 600` and `--val-epoch 5` defaults still apply; set
`--val-start 0` to validate from the first epoch. Mix is loaded as supplied,
not generated by the DataLoader. Keep original-image identities disjoint
between training and validation, including clean/hazy variants.
