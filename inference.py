import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm

import models


class SegmentationFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path = self.imgs[index][0]
        sample = self.loader(path)
        height, width = sample.size[1], sample.size[0]

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, path, height, width


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--nb_classes', type=int, default=50)
    parser.add_argument('--mode',
                        type=str,
                        required=True,
                        help='validation or test',
                        choices=['validation', 'test'])
    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help='the path to save segmentation masks')
    parser.add_argument('--data_path',
                        type=str,
                        default=None,
                        help='path to imagenetS dataset')
    parser.add_argument('--finetune',
                        type=str,
                        default=None,
                        help='the model checkpoint file')
    parser.add_argument('--pretrained_rfnext',
                        default='',
                        help='pretrained weights for RF-Next')
    parser.add_argument('--model',
                        default='vit_small_patch16',
                        help='model architecture')
    parser.add_argument('--patch_size',
                        type=int,
                        default=4,
                        help='For convnext/rfconvnext, the numnber of output channels is '
                        'nb_classes * patch_size * patch_size.'
                        'https://arxiv.org/pdf/2111.06377.pdf')
    parser.add_argument(
        '--max_res',
        default=1000,
        type=int,
        help='Maximum resolution for evaluation. 0 for disable.')
    parser.add_argument('--method',
                        default='example submission',
                        help='Method name in method description file(.txt).')
    parser.add_argument('--train_data',
                        default='null',
                        help='Training data in method description file(.txt).')
    parser.add_argument(
        '--train_scheme',
        default='null',
        help='Training scheme in method description file(.txt), \
            e.g., SSL, Sup, SSL+Sup.')
    parser.add_argument(
        '--link',
        default='null',
        help='Paper/project link in method description file(.txt).')
    parser.add_argument(
        '--description',
        default='null',
        help='Method description in method description file(.txt).')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # build model
    model = models.__dict__[args.model](args)
    model = model.cuda()
    model.eval()

    # load checkpoints
    checkpoint = torch.load(args.finetune)['model']
    model.load_state_dict(checkpoint, strict=True)
    # build the dataloader
    dataset_path = os.path.join(args.data_path, args.mode)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = SegmentationFolder(root=dataset_path,
                                 transform=transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.ToTensor(),
                                     normalize,
                                 ]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             num_workers=16,
                                             pin_memory=True)

    output_dir = os.path.join(args.output_dir, args.mode)

    for images, path, height, width in tqdm(dataloader):
        path = path[0]
        cate = path.split('/')[-2]
        name = path.split('/')[-1].split('.')[0]
        if not os.path.exists(os.path.join(output_dir, cate)):
            os.makedirs(os.path.join(output_dir, cate))

        with torch.no_grad():
            H = height.item()
            W = width.item()

            output = model.forward(images.cuda())

            if (H > W and H * W > args.max_res * args.max_res
                    and args.max_res > 0):
                output = F.interpolate(
                    output, (args.max_res, int(args.max_res * W / H)),
                    mode='bilinear',
                    align_corners=False)
                output = torch.argmax(output, dim=1, keepdim=True)
                output = F.interpolate(output.float(), (H, W),
                                       mode='nearest').long()
            elif (H <= W and H * W > args.max_res * args.max_res
                  and args.max_res > 0):
                output = F.interpolate(
                    output, (int(args.max_res * H / W), args.max_res),
                    mode='bilinear',
                    align_corners=False)
                output = torch.argmax(output, dim=1, keepdim=True)
                output = F.interpolate(output.float(), (H, W),
                                       mode='nearest').long()
            else:
                output = F.interpolate(output, (H, W),
                                       mode='bilinear',
                                       align_corners=False)
                output = torch.argmax(output, dim=1, keepdim=True)
            output = output.squeeze()

            res = torch.zeros(size=(output.shape[0], output.shape[1], 3))
            res[:, :, 0] = output % 256
            res[:, :, 1] = output // 256
            res = res.cpu().numpy()

            res = Image.fromarray(res.astype(np.uint8))
            res.save(os.path.join(output_dir, cate, name + '.png'))

    if args.mode == 'test':
        method = 'Method name: {}\n'.format(
                args.method) + \
            'Training data: {}\nTraining scheme: {}\n'.format(
                args.train_data, args.train_scheme) + \
            'Networks: {}\nPaper/Project link: {}\n'.format(
                args.model, args.link) + \
            'Method description: {}'.format(
                args.description)
        with open(os.path.join(output_dir, 'method.txt'), 'w') as f:
            f.write(method)

        # zip for submission
        shutil.make_archive(os.path.join(args.output_dir, args.mode),
                            'zip',
                            root_dir=output_dir)


if __name__ == '__main__':
    main()
