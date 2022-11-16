# Model ZOO for Semi-Supervised Learning on ImageNet-S

[Finetuning with ViT](#1)

[Finetuning with ResNet](#2)

[Finetuning with RF-ConvNext](#3)


<div id="1"></div>

## Finetuning with ViT

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Arch</th>
<th valign="bottom">Pretraining epochs</th>
<th valign="bottom">Pretraining mode</th>
<th valign="bottom">val</th>
<th valign="bottom">test</th>
<th valign="bottom">Pretrained</th>
<th valign="bottom">Finetuned</th>
<!-- TABLE BODY -->
<tr>
<td align="center"><a href="https://arxiv.org/abs/2111.06377">MAE</a></td>
<td align="center">ViT-B/16</td>
<td align="center">1600</td>
<td align="center">SSL</td>
<td align="center">38.3</td>
<td align="center">37.0</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth">model</a></td>
<td align="center"><a href="https://github.com/LUSSeg/ImageNetSegModel/releases/download/vit/imagenets_ssl_mae_vit_base.pth">model</a></td>
</tr>
<td align="center"><a href="https://arxiv.org/abs/2111.06377">MAE</a></td>
<td align="center">ViT-B/16</td>
<td align="center">1600</td>
<td align="center">SSL+Sup</td>
<td align="center">61.0</td>
<td align="center">60.2</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">model</a></td>
<td align="center"><a href="https://github.com/LUSSeg/ImageNetSegModel/releases/download/vit/imagenets_ssl-sup_mae_vit_base.pth">model</a></td>
</tr>
</tr>
<td align="center"><a href="https://arxiv.org/abs/2206.05184">SERE</a></td>
<td align="center">ViT-S/16</td>
<td align="center">100</td>
<td align="center">SSL</td>
<td align="center">41.0</td>
<td align="center">40.2</td>
<td align="center"><a href="https://github.com/LUSSeg/ImageNetSegModel/releases/download/vit/sere_pretrained_vit_small_ep100.pth">model</a></td>
<td align="center"><a href="https://github.com/LUSSeg/ImageNetSegModel/releases/download/vit/imagenets_ssl_sere_vit_small.pth">model</a></td>
</tr>
<td align="center"><a href="https://arxiv.org/abs/2206.05184">SERE</a></td>
<td align="center">ViT-S/16</td>
<td align="center">100</td>
<td align="center">SSL+Sup</td>
<td align="center">58.9</td>
<td align="center">57.8</td>
<td align="center"><a href="https://github.com/LUSSeg/ImageNetSegModel/releases/download/vit/sere_finetuned_vit_small_ep100.pth">model</a></td>
<td align="center"><a href="https://github.com/LUSSeg/ImageNetSegModel/releases/download/vit/imagenets_ssl-sup_sere_vit_small.pth">model</a></td>
</tr>
</tbody></table>

### <a href="https://arxiv.org/abs/2111.06377">Masked Autoencoders Are Scalable Vision Learners (MAE)</a>

<details>
  <summary>Command for SSL+Sup</summary>

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_segfinetune.py \
--accum_iter 1 \
--batch_size 32 \
--model vit_base_patch16 \
--finetune mae_finetuned_vit_base.pth \
--epochs 100 \
--nb_classes 920 \
--blr 1e-4 --layer_decay 0.40 \
--weight_decay 0.05 --drop_path 0.1  \
--data_path ${IMAGENETS_DIR} \
--output_dir ${OUTPATH} \
--dist_eval
```

</details>

<details>
  <summary>Command for SSL</summary>

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_segfinetune.py \
--accum_iter 1 \
--batch_size 32 \
--model vit_base_patch16 \
--finetune mae_pretrain_vit_base.pth \
--epochs 100 \
--nb_classes 920 \
--blr 5e-4 --layer_decay 0.60 \
--weight_decay 0.05 --drop_path 0.1  \
--data_path ${IMAGENETS_DIR} \
--output_dir ${OUTPATH} \
--dist_eval
```

</details>

### <a href="https://arxiv.org/abs/2206.05184">SERE: Exploring Feature Self-relation for Self-supervised Transformer </a>

<details>
  <summary>Command for SSL+Sup</summary>

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_segfinetune.py \
--accum_iter 1 \
--batch_size 32 \
--model vit_small_patch16 \
--finetune sere_finetuned_vit_small_ep100.pth \
--epochs 100 \
--nb_classes 920 \
--blr 5e-4 --layer_decay 0.50 \
--weight_decay 0.05 --drop_path 0.1  \
--data_path ${IMAGENETS_DIR} \
--output_dir ${OUTPATH} \
--dist_eval
```

</details>

<details>
  <summary>Command for SSL</summary>

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_segfinetune.py \
--accum_iter 1 \
--batch_size 32 \
--model vit_small_patch16 \
--finetune sere_pretrained_vit_small_ep100.pth \
--epochs 100 \
--nb_classes 920 \
--blr 5e-4 --layer_decay 0.50 \
--weight_decay 0.05 --drop_path 0.1  \
--data_path ${IMAGENETS_DIR} \
--output_dir ${OUTPATH} \
--dist_eval
```
</details>


<div id="2"></div>

## Finetuning with ResNet
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Arch</th>
<th valign="bottom">Pretraining epochs</th>
<th valign="bottom">Pretraining mode</th>
<th valign="bottom">val</th>
<th valign="bottom">test</th>
<th valign="bottom">Pretrained</th>
<th valign="bottom">Finetuned</th>
<!-- TABLE BODY -->
<tr>
<td align="center"><a href="https://arxiv.org/abs/2106.03149">PASS</a></td>
<td align="center">ResNet-50 D32</td>
<td align="center">100</td>
<td align="center">SSL</td>
<td align="center">21.0</td>
<td align="center">20.3</td>
<td align="center"><a href="https://github.com/LUSSeg/PASS/releases/download/pass/pass919_pretrained.pth.tar">model</a></td>
<td align="center"><a href="https://github.com/LUSSeg/ImageNetSegModel/releases/download/pass/imagenets_ssl_pass_resnet50_d32.pth">model</a></td>
</tr>
<tr>
<td align="center"><a href="https://arxiv.org/abs/2106.03149">PASS</a></td>
<td align="center">ResNet-50 D16</td>
<td align="center">100</td>
<td align="center">SSL</td>
<td align="center">21.6</td>
<td align="center">20.8</td>
<td align="center"><a href="https://github.com/LUSSeg/PASS/releases/download/pass/pass919_pretrained.pth.tar">model</a></td>
<td align="center"><a href="https://github.com/LUSSeg/ImageNetSegModel/releases/download/pass/imagenets_ssl_pass_resnet50_d16.pth">model</a></td>
</tr>
</tbody></table>

`D16` means the output stride is 16 with dilation=2 in the last stage.

### <a href="https://arxiv.org/abs/2206.05184">Large-scale Unsupervised Semantic Segmentation (PASS)</a>
<details>
  <summary>Command for SSL (ResNet-50 D32)</summary>

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_segfinetune.py \
--accum_iter 1 \
--batch_size 32 \
--model resnet50 \
--finetune pass919_pretrained.pth.tar \
--epochs 100 \
--nb_classes 920 \
--blr 5e-4 --layer_decay 0.4 \
--weight_decay 0.0005 \
--data_path ${IMAGENETS_DIR} \
--output_dir ${OUTPATH} \
--dist_eval
```
</details>

<details>
  <summary>Command for SSL (ResNet-50 D16)</summary>

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_segfinetune.py \
--accum_iter 1 \
--batch_size 32 \
--model resnet50_d16 \
--finetune pass919_pretrained.pth.tar \
--epochs 100 \
--nb_classes 920 \
--blr 5e-4 --layer_decay 0.45 \
--weight_decay 0.0005 \
--data_path ${IMAGENETS_DIR} \
--output_dir ${OUTPATH} \
--dist_eval
```
</details>


<div id="3"></div>

## Finetuning with RF-ConvNeXt

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Arch</th>
<th valign="bottom">Pretraining epochs</th>
<th valign="bottom">RF-Next mode</th>
<th valign="bottom">val</th>
<th valign="bottom">test</th>
<th valign="bottom">Pretrained</th>
<th valign="bottom">Finetuned</th>
<!-- TABLE BODY -->
<tr>
<td align="center"><a href="https://arxiv.org/abs/2201.03545">ConvNeXt-T</a></td>
<td align="center">300</td>
<td align="center">-</td>
<td align="center">48.7</td>
<td align="center">48.8</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth">model</a></td>
<td align="center"><a href="https://github.com/LUSSeg/ImageNetSegModel/releases/download/rfconvnext_tiny/imagenets_sup_convnext_tiny.pth">model</a></td>
</tr>
<tr>
<td align="center"><a href="https://arxiv.org/abs/2206.06637">RF-ConvNeXt-T</a></td>
<td align="center">300</td>
<td align="center">rfsingle</td>
<td align="center">50.7</td>
<td align="center">50.5</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth">model</a></td>
<td align="center"><a href="https://github.com/LUSSeg/ImageNetSegModel/releases/download/rfconvnext_tiny/imagenets_sup_rfnext_convnext_tiny_rfsingle.pth">model</a></td>
</tr>
<tr>
<td align="center"><a href="https://arxiv.org/abs/2206.06637">RF-ConvNeXt-T</a></td>
<td align="center">300</td>
<td align="center">rfmultiple</td>
<td align="center">50.8</td>
<td align="center">50.5</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth">model</a></td>
<td align="center"><a href="https://github.com/LUSSeg/ImageNetSegModel/releases/download/rfconvnext_tiny/imagenets_sup_rfnext_convnext_tiny_rfmultiple.pth">model</a></td>
</tr>
<tr>
<td align="center"><a href="https://arxiv.org/abs/2206.06637">RF-ConvNeXt-T</a></td>
<td align="center">300</td>
<td align="center">rfmerge</td>
<td align="center">51.3</td>
<td align="center">51.1</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth">model</a></td>
<td align="center"><a href="https://github.com/LUSSeg/ImageNetSegModel/releases/download/rfconvnext_tiny/imagenets_sup_rfnext_convnext_tiny_rfmerge.pth">model</a></td>
</tr>
</tbody></table>

<details>
  <summary>Command for ConvNeXt-T</summary>
  
```shell
python -m torch.distributed.launch --nproc_per_node=8 main_segfinetune.py \
--accum_iter 1 \
--batch_size 32 \
--model convnext_tiny \
--patch_size 4 \
--finetune convnext_tiny_1k_224_ema.pth \
--epochs 100 \
--nb_classes 920 \
--blr 2.5e-4 --layer_decay 0.6 \
--weight_decay 0.05 --drop_path 0.2  \
--data_path ${IMAGENETS_DIR} \
--output_dir ${OUTPATH} \
--dist_eval
```
</details>

Before training RF-ConvNext, 
please search dilation rates with the mode of rfsearch. 

For rfmultiple and rfsingle, please set `pretrained_rfnext` 
as the weights trained in rfsearch. 

For rfmerge, we initilize the model with weights in rfmultiple and only finetune `seg_norm`, `seg_head` and `rfconvs` whose dilate rates are changed. 
The othe parts of the network are freezed.
Please set `pretrained_rfnext` 
as the weights trained in rfmutilple. 

**Note that this freezing operation in rfmerge may be not required for other tasks.**

<details>
  <summary>Command for RF-ConvNeXt-T (rfsearch)</summary>
  
```shell
python -m torch.distributed.launch --nproc_per_node=8 main_segfinetune.py \
--accum_iter 1 \
--batch_size 32 \
--model rfconvnext_tiny_rfsearch \
--patch_size 4 \
--finetune convnext_tiny_1k_224_ema.pth \
--epochs 100 \
--nb_classes 920 \
--blr 2.5e-4 --layer_decay 0.6 0.9 --layer_multiplier 1.0 10.0 \
--weight_decay 0.05 --drop_path 0.2  \
--data_path ${IMAGENETS_DIR} \
--output_dir ${OUTPATH} \
--dist_eval
```
</details>

<details>
  <summary>Command for RF-ConvNeXt-T (rfsingle)</summary>

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_segfinetune.py \
--accum_iter 1 \
--batch_size 32 \
--model rfconvnext_tiny_rfsingle \
--patch_size 4 \
--finetune convnext_tiny_1k_224_ema.pth \
--pretrained_rfnext ${OUTPATH_OF_RFSEARCH}/checkpoint-99.pth \
--epochs 100 \
--nb_classes 920 \
--blr 2.5e-4 --layer_decay 0.6 0.9 --layer_multiplier 1.0 10.0 \
--weight_decay 0.05 --drop_path 0.2  \
--data_path ${IMAGENETS_DIR} \
--output_dir ${OUTPATH} \
--dist_eval

python inference.py --model rfconvnext_tiny_rfsingle \
--patch_size 4 \
--nb_classes 920 \
--output_dir ${OUTPATH}/predictions \
--data_path ${IMAGENETS_DIR} \
--pretrained_rfnext ${OUTPATH_OF_RFSEARCH}/checkpoint-99.pth \
--finetune ${OUTPATH}/checkpoint-99.pth \
--mode validation
```
</details>

<details>
  <summary>Command for RF-ConvNeXt-T (rfmultiple)</summary>

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_segfinetune.py \
--accum_iter 1 \
--batch_size 32 \
--model rfconvnext_tiny_rfmultiple \
--patch_size 4 \
--finetune convnext_tiny_1k_224_ema.pth \
--pretrained_rfnext ${OUTPATH_OF_RFSEARCH}/checkpoint-99.pth \
--epochs 100 \
--nb_classes 920 \
--blr 2.5e-4 --layer_decay 0.55 0.9 --layer_multiplier 1.0 10.0 \
--weight_decay 0.05 --drop_path 0.1  \
--data_path ${IMAGENETS_DIR} \
--output_dir ${OUTPATH} \
--dist_eval

python inference.py --model rfconvnext_tiny_rfmultiple \
--patch_size 4 \
--nb_classes 920 \
--output_dir ${OUTPATH}/predictions \
--data_path ${IMAGENETS_DIR} \
--pretrained_rfnext ${OUTPATH_OF_RFSEARCH}/checkpoint-99.pth \
--finetune ${OUTPATH}/checkpoint-99.pth \
--mode validation
```
</details>


<details>
  <summary>Command for RF-ConvNeXt-T (rfmerge)</summary>

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_segfinetune.py \
--accum_iter 1 \
--batch_size 32 \
--model rfconvnext_tiny_rfmerge \
--patch_size 4 \
--pretrained_rfnext ${OUTPATH_OF_RFMULTIPLE}/checkpoint-99.pth \
--epochs 100 \
--nb_classes 920 \
--blr 2.5e-4 --layer_decay 0.55 1.0 --layer_multiplier 1.0 10.0 \
--weight_decay 0.05 --drop_path 0.2  \
--data_path ${IMAGENETS_DIR} \
--output_dir ${OUTPATH} \
--dist_eval

python inference.py --model rfconvnext_tiny_rfmerge \
--patch_size 4 \
--nb_classes 920 \
--output_dir ${OUTPATH}/predictions \
--data_path ${IMAGENETS_DIR} \
--pretrained_rfnext ${OUTPATH_OF_RFMULTIPLE}/checkpoint-99.pth \
--finetune ${OUTPATH}/checkpoint-99.pth \
--mode validation
```
</details>
