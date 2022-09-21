# Semi-supervised Semantic Segmentation on the ImageNet-S dataset

This repo provides the code of semi-supervised training of large-scale semantic segmentation on the ImageNet-S dataset.

## About ImageNet-S
Based on the ImageNet dataset, the ImageNet-S dataset has 1.2 million training images and 50k high-quality semantic segmentation annotations to 
support unsupervised/semi-supervised semantic segmentation on the ImageNet dataset. ImageNet-S dataset is available on [ImageNet-S](https://github.com/LUSSeg/ImageNet-S). More details about the dataset please refer to the [project page](https://LUSSeg.github.io/) or [paper link](https://arxiv.org/abs/2106.03149).



## Usage
- Semi-supervised finetuning with pre-trained checkpoints
```
python -m torch.distributed.launch --nproc_per_node=8 main_segfinetune.py \
--accum_iter 1 \
--batch_size 32 \
--model vit_small_patch16 \
--finetune ${PRETRAIN_CHKPT} \
--epochs 100 \
--nb_classes 920 | 301 | 51 \
--blr 5e-4 --layer_decay 0.50 \
--weight_decay 0.05 --drop_path 0.1  \
--data_path ${IMAGENETS_DIR} \
--output_dir ${OUTPATH} \
--dist_eval
```
- Get the zip file for testing set. You can submit it to our [online server](https://lusseg.github.io/).
```
python inference.py --model vit_small_patch16 \
--nb_classes 920 | 301 | 51 \
--output_dir ${OUTPATH}/predictions \
--data_path ${IMAGENETS_DIR} \
--finetune ${OUTPATH}/checkpoint-99.pth \
--mode validation | test
```

## Model Zoo
**[Model Zoo](MODEL_ZOO.md)**:
We provide a model zoo to record the trend of semi-supervised semantic segmentation on the ImageNet-S dataset.
For now, this repo supports ViT, and more backbones and pretrained models will be added.
Please open a pull request if you want to update your new results.

Supported networks: ViT

Supported pretrain: MAE, SERE

## Citation
```
@article{gao2021luss,
  title={Large-scale Unsupervised Semantic Segmentation},
  author={Gao, Shanghua and Li, Zhong-Yu and Yang, Ming-Hsuan and Cheng, Ming-Ming and Han, Junwei and Torr, Philip},
  journal={arXiv preprint arXiv:2106.03149},
  year={2021}
}
```

## Acknowledgement

This codebase is build based on the [MAE codebase](https://github.com/facebookresearch/mae).
