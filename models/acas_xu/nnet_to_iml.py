from nnet import NNet
import numpy as np
import math

import sys
import os
from pathlib import Path


def write_matrix(file, m):
    file.write('[\n')
    for line in m:
        file.write('  [')
        for x in line:
            file.write('{:.5f}; '.format(x))
        file.write(']; \n')
    file.write(']; \n')


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return int(math.floor(n * multiplier + 0.5) / multiplier)


def write_matrix_quantized(file, m):
    file.write('[\n')
    for line in m:
        file.write('  [')
        for x in line:
            file.write('{:d}; '.format(round_half_up(x)))
        file.write(']; \n')
    file.write('] \n')


def nnet_to_iml_weights(nnet_file, dst_dir=None, quantized=False):
    """ Convert .nnet file to .iml list representation """
    nnet_file = Path(nnet_file)
    if dst_dir is None:
        dst_dir = nnet_file.parent

    iml_file = dst_dir / (nnet_file.stem + '.iml')
    to_iml = write_matrix_quantized if quantized else write_matrix
    n = NNet(nnet_file)
    file = open(iml_file, 'w')

    file.write('module Weights = struct\n')
    for w, b, i in zip(n.weights, n.biases, range(len(n.weights))):
        w = np.insert(w, 0, b, axis=1)  # add bias at start of weights line
        file.write(f'\nlet layer{i} = ')
        to_iml(file, w)

    file.write('end')
    file.close()
    return iml_file


if __name__ == '__main__':
    src_dir = Path('./networks_nnet')
    dst_dir = Path('./networks_iml')
    for file in os.listdir(src_dir):
        src_path = Path(src_dir) / file
        out_path = nnet_to_iml_weights(src_path, dst_dir=dst_dir, quantized=False)
        print(f'Wrote iml to {out_path}.')
