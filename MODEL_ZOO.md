# Model ZOO for Semi-Supervised Learning on ImageNet-S

## Finetuning with ViTs

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
