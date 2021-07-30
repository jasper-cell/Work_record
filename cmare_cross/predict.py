import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

def predict_img(net,
                net_zc,
                net_cancer,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    if net_zc is not None:
        net_zc.eval()
    if net_cancer is not None:
        net_cancer.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))  # 对需要预测的图像做同样的预处理工作

    img = img.unsqueeze(0)  # 增加一个对应的batch dim
    img = img.to(device=device, dtype=torch.float32)  # 将预测图像迁移到GPU上
    # img = img.float()

    with torch.no_grad():
        output = net(img)  # 对输入图像进行前向传播
        if net_zc is not None:
            output_zc = net_zc(img)
        if net_cancer is not None:
            output_cancer = net_cancer(img)

        # 根据不同的类别来选择对应的损失函数
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
            if net_zc is not None:
                probs_zc = F.softmax(output_zc, dim=1)
            if net_cancer is not None:
                probs_cancer = F.softmax(output_cancer, dim=1)
        else:
            probs = torch.sigmoid(output)
            if net_zc is not None:
                probs_zc = torch.sigmoid(output_zc)
            if net_cancer is not None:
                probs_cancer = torch.sigmoid(output_cancer)

        probs = probs.squeeze(0)  # 对batch轴进行压缩
        if net_zc is not None:
            probs_zc = probs_zc.squeeze(0)
        if net_cancer is not None:
            probs_cancer = probs_cancer.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),  # 此处应该是一个还原的过程
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())  # 迁移到cpu并对其进行transforms
        if net_zc is not None:
            probs_zc = tf(probs_zc.cpu())
        if net_cancer is not None:
            probs_cancer = tf(probs_cancer.cpu())

        full_mask = probs.squeeze().cpu().numpy()  # 转为numpy数组
        if net_zc is not None:
            full_mask_zc = probs_zc.squeeze().cpu().numpy()
        if net_cancer is not None:
            full_mask_cancer = probs_cancer.squeeze().cpu().numpy()

    if net_zc is not None and net_cancer is not None:
        return full_mask > out_threshold, full_mask_zc > out_threshold, full_mask_cancer > out_threshold
    elif net_zc is not None:
        return full_mask > out_threshold, full_mask_zc > out_threshold
    else:
        return full_mask > out_threshold, None

def predict_img_multi(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))  # 对需要预测的图像做同样的预处理工作

    img = img.unsqueeze(0)  # 增加一个对应的batch dim
    img = img.to(device=device, dtype=torch.float32)  # 将预测图像迁移到GPU上

    with torch.no_grad():
        output = net(img)  # 对输入图像进行前向传播

        # 根据不同的类别来选择对应的损失函数
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)  # 对batch轴进行压缩

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),  # 此处应该是一个还原的过程
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())  # 迁移到cpu并对其进行transforms
        full_mask = probs.squeeze().cpu().numpy()  # 转为numpy数组

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='./weights/epoch_26_64c.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--model_zc', '-mzc', default='./weights/epoch_20_z64c.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument("--model_cancer", '-mcancer', default='./DaTi_weights/epoch_9.pth',
                        metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', default='./medical_imgs/lung1.png',
                        help='filenames of input images', required=False)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', default="./result_of_output.jpg",
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)
    parser.add_argument("--width", "-w", default=2, required=False)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)


    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        mask = predict_img(net=net,net_zc=None,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
