import os, sys
import cv2
import argparse

import numpy as np

from poisson_utils import Poisson


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='poisson_tasks_img/img/archive_2/4/source.png ')
    parser.add_argument('--dst', type=str, default='poisson_tasks_img/img/archive_2/4/target.png')
    parser.add_argument('--mask', type=str, default='poisson_tasks_img/img/archive_2/4/mask.png')
    parser.add_argument('--output', type=str, default='poisson_tasks_img/img/archive_2/4/output.png')

    parser.add_argument('--method', type=str, default='seamlessClone')  # seamlessClonecolorChangeilluminationChange
    return parser


if __name__ == '__main__':
    args = set_args().parse_args()
    src = cv2.imread(args.src)
    if hasattr(args, 'dst') and args.dst is not None:
        dst = cv2.imread(args.dst)
    if hasattr(args, 'mask') and args.mask is not None:
        mask3 = cv2.imread(args.mask)
        mask = mask3[:, :, 0]

    method = args.method


    if method == 'seamlessClone':
        output = Poisson().seamlessClone(src, dst, mask)
    elif method == 'mixedClone':
        output = Poisson().mixedClone(src, dst, mask)
    elif method == "colorChange":
        src_converted = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        output = Poisson().colorChange(src_converted, mask, 0.5, 0.5, 1.5)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    elif method == "illuminationChange":
        output = Poisson().illuminationChange(src, mask, 0.1, 0.2)
    elif method == 'textureFlatten':
        output = Poisson().textureFlatten(src, mask, 50, 60)
    elif method == 'tiling':
        result = Poisson().tiling(src)
        output = np.tile(result, (2, 2, 1))


    else:
        print('method error')
        sys.exit(0)
    cv2.imwrite(args.output, output)
    cv2.imshow('output', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
