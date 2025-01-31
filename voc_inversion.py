# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2020 paper
# Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion
# Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek
# Hoiem, Niraj K. Jha, and Jan Kautz
# --------------------------------------------------------

from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse
import torch

import random
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

import torchvision.models as models
from utils.utils import load_model_pytorch
from models.build_BiSeNet import BiSeNet
from models.segmentation_module_BiSeNet import IncrementalSegmentationBiSeNet

from utils.utils import CustomPooling


random.seed(0)

def validate_one(input, target, model):
    """Perform validation on the validation set"""

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)

        customPooling = CustomPooling()
        output = customPooling(output)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


def run(args):
    torch.manual_seed(args.local_rank)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # until now, we only have the net -> it is not pretrained
    if args.arch_name == "resnet50v15":
        from models.resnetv15 import build_resnet
        net = build_resnet("resnet34", "classic")
    else:
        print("loading torchvision model for inversion with the name: {}".format(args.arch_name))

        checkpoint_teacher = torch.load(args.teacher_path)
        checkpoint_teacher = checkpoint_teacher['model_state']

        head = BiSeNet("resnet50")
        body = "resnet50"
        net = IncrementalSegmentationBiSeNet(body, head, classes=[16], fusion_mode="mean")

        net.load_state_dict(checkpoint_teacher, strict=False)


    net = net.to(device)

    print('==> Resuming from checkpoint..')

    if args.arch_name == "resnet50v15":
        path_to_model = "./models/resnet50v15/model_best.pth.tar"
        load_model_pytorch(net, path_to_model, gpu_n=torch.cuda.current_device())

    net.to(device)
    net.eval()

    # reserved to compute test accuracy on generated images by different networks
    net_verifier = None
    if args.verifier and args.adi_scale == 0:
        # if multiple GPUs are used then we can change code to load different verifiers to different GPUs
        if args.local_rank == 0:
            print("loading verifier: ", args.verifier_arch)
            # here we should load our pre trained network on the VOC
            net_verifier = models.__dict__[args.verifier_arch](pretrained=False).to(device)
            net_verifier.eval()


    if args.adi_scale != 0.0:
        student_arch = "resnet18"

        checkpoint_verifier = torch.load(args.student_path)

        checkpoint_verifier = checkpoint_verifier['model_state']
        head = BiSeNet("resnet18")
        body = "resnet18"

        net_verifier = IncrementalSegmentationBiSeNet(body, head, classes=[16], fusion_mode="mean")
        net_verifier.load_state_dict(checkpoint_verifier, strict=False)

        net_verifier = net_verifier.to(device)
        net_verifier.train()


    from deepinversion import DeepInversionClass

    exp_name = args.exp_name
    # final images will be stored here:
    adi_data_path = "./final_images/%s" % exp_name
    # temporal data and generations will be stored here
    exp_name = "generations/%s" % exp_name

    args.iterations = 2000
    args.start_noise = True

    args.resolution = 224
    bs = args.bs
    jitter = 30

    # default settings
    parameters = dict()
    parameters["resolution"] = 256
    parameters["random_label"] = False
    parameters["start_noise"] = True
    parameters["detach_student"] = False
    parameters["do_flip"] = True

    parameters["do_flip"] = args.do_flip
    parameters["random_label"] = args.random_label
    parameters["store_best_images"] = args.store_best_images

    criterion = nn.CrossEntropyLoss()

    coefficients = dict()
    coefficients["r_feature"] = args.r_feature
    coefficients["first_bn_multiplier"] = args.first_bn_multiplier
    coefficients["tv_l1"] = args.tv_l1
    coefficients["tv_l2"] = args.tv_l2
    coefficients["l2"] = args.l2
    coefficients["lr"] = args.lr
    coefficients["main_loss_multiplier"] = args.main_loss_multiplier
    coefficients["adi_scale"] = args.adi_scale

    network_output_function = lambda x: x

    if args.verifier:
        hook_for_display = lambda x, y: validate_one(x, y, net_verifier)
    else:
        hook_for_display = None

    DeepInversionEngine = DeepInversionClass(net_teacher=net,
                                             final_data_path=adi_data_path,
                                             path=exp_name,
                                             parameters=parameters,
                                             setting_id=args.setting_id,
                                             bs=bs,
                                             use_fp16=args.fp16,
                                             jitter=jitter,
                                             criterion=criterion,
                                             coefficients=coefficients,
                                             network_output_function=network_output_function,
                                             hook_for_display=hook_for_display)
    net_student = None
    if args.adi_scale != 0:
        net_student = net_verifier
    DeepInversionEngine.generate_batch(net_student=net_student, n_batches=args.n_batches)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-nb', '--n_batches', type=int, default=1, help='Number of batches to generate for each run')
    parser.add_argument('-s', '--worldsize', type=int, default=1, help='Number of processes participating in the job.')
    parser.add_argument('--local_rank', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--adi_scale', type=float, default=0.0, help='Coefficient for Adaptive Deep Inversion')
    parser.add_argument('--no-cuda', action='store_true')

    parser.add_argument('--epochs', default=20000, type=int, help='batch size')
    parser.add_argument('--setting_id', default=0, type=int,
                        help='settings for optimization: 0 - multi resolution, 1 - 2k iterations, 2 - 20k iterations')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--jitter', default=30, type=int, help='batch size')
    parser.add_argument('--comment', default='', type=str, help='batch size')
    parser.add_argument('--arch_name', default='resnet50', type=str, help='model name from torchvision or resnet50v15')

    parser.add_argument('--fp16', action='store_true', help='use FP16 for optimization')
    parser.add_argument('--exp_name', type=str, default='test', help='where to store experimental data')

    parser.add_argument('--verifier', action='store_true', help='evaluate batch with another model')
    parser.add_argument('--verifier_arch', type=str, default='mobilenet_v2',
                        help="arch name from torchvision models to act as a verifier")

    parser.add_argument('--do_flip', action='store_true', help='apply flip during model inversion')
    parser.add_argument('--random_label', action='store_true', help='generate random label for optimization')
    parser.add_argument('--r_feature', type=float, default=0.05,
                        help='coefficient for feature distribution regularization')
    parser.add_argument('--first_bn_multiplier', type=float, default=10.,
                        help='additional multiplier on first bn layer of R_feature')
    parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss')
    parser.add_argument('--tv_l2', type=float, default=0.0001, help='coefficient for total variation L2 loss')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate for optimization')
    parser.add_argument('--l2', type=float, default=0.00001, help='l2 loss on the image')
    parser.add_argument('--main_loss_multiplier', type=float, default=1.0,
                        help='coefficient for the main loss in optimization')
    parser.add_argument('--store_best_images', action='store_true', help='save best images as separate files')

    # new arguments defined us for the sake of simplicity
    # >> default teacher : BiSeNet with a ResNet50 as backbone
    # >> default student : BiSeNet with a ResNet18 as backbone
    parser.add_argument('--teacher_path', type=str, default="/content/drive/MyDrive/step-0-resnet50.pth",
                        help="Path of the pre trained teacher path")
    parser.add_argument('--student_path', type=str, default="/content/drive/MyDrive/step-0-resnet18.pth",
                        help="Path of the pre trained student path")


    args = parser.parse_args()
    print(args)

    torch.backends.cudnn.benchmark = True
    run(args)


if __name__ == '__main__':
    main()
