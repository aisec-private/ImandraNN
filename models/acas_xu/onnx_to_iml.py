import argparse
from email.policy import default
import os
import sys

import numpy as np
import onnx
from numpy import random as rand
from onnx import numpy_helper
from onnxruntime.quantization import quantize_static, QuantFormat
from onnxruntime.quantization.calibrate import CalibrationDataReader

from pathlib import Path

rand.seed(42)


class AcasxuDataReader(CalibrationDataReader):
    '''
    ACAS Xu mock data generator for network quantization by onnxruntime library
    '''
    def __init__(self):
        self.inputs = iter([{'input': self.make_input()} for _ in range(100)])

    def make_input(self):
        rho = rand.uniform(0, 60000)
        theta = rand.uniform(-np.pi, np.pi)
        psi = rand.uniform(-np.pi, np.pi)
        v_own = rand.normal(200, 100)
        v_int = rand.normal(180, 100)
        out = np.array([rho, theta, psi, v_own, v_int], dtype='float32')
        out = out.reshape((1, 1, 1, 5))
        return out

    def get_next(self):
        return next(self.inputs, None)


#
# Functions to write matrices to file according to:
# - formalisation: L = lists (of lists), F = function
# - type of values: I = integers, R = reals
#

def write_matrix_LI(file, m, var_name):
    '''
    Formalization: lists
    Value type: integers
    '''
    file.write(f'let {var_name} = [\n')
    for line in m:
        file.write('  [ ')
        for x in line:
            file.write(f'{x}; ')
        file.write(' ];\n')
    file.write(']\n')
 

def write_matrix_LR(file, m, var_name):
    '''
    Formalization: lists
    Value type: reals
    '''
    file.write(f'let {var_name} = [\n')
    for line in m:
        file.write('  [ ')
        for x in line:
            file.write(f'{x:.5f}; ')
        file.write(' ];\n')
    file.write(']\n')


def write_matrix_FI(file, m, var_name):
    '''
    Formalization: function
    Value type: integers
    '''
    file.write(f'let {var_name}_map =\n')
    for (i, line) in enumerate(m):
        for (j, el) in enumerate(line):
            if el != 0:
                file.write(f'\tMap.add ({i},{j}) ({m[i,j]}) @@\n')
    file.write('\tMap.const 0\n')
    file.write(
        f'''\nlet {var_name} = FC.fc FC.relu (
    function
        Rows -> {m.shape[0]}
        | Cols -> {m.shape[1]}
        | Value (i,j) -> Map.get (i,j) {var_name}_map
    )\n
    ''' 
    )


def write_matrix_FR(file, m, var_name):
    '''
    Formalization: function
    Value type: reals
    '''
    file.write(f'let {var_name}_map =\n')
    for (i, line) in enumerate(m):
        for (j, el) in enumerate(line):
            if el != 0:
                file.write(f'\tMap.add ({i}.,{j}.) ({m[i,j]:.5f}) @@\n')
    file.write('\tMap.const 0.\n')
    file.write(
        f'''\nlet {var_name} = FC.fc FC.relu (
    function
        Rows -> {m.shape[0]}.
        | Cols -> {m.shape[1]}.
        | Value (i,j) -> (Map.get (i,j) {var_name}_map)
    )\n
    ''' 
    )


def write_matrix(file, m, quantize, form, var_name):
    if quantize and form == 'lists':
        write_matrix_LI(file, m)
    elif quantize and form == 'func':
        write_matrix_FI(file, m, var_name)
    elif not quantize and form == 'lists':
        write_matrix_LR(file, m)
    elif not quantize and form == 'func':
        write_matrix_FR(file, m, var_name)


def write_network(model, output_file, form='func', **kwargs):
    with open(output_file, 'w+') as of:
        of.write('module Weights = struct\n')
        for (i, weights) in enumerate(model):
            write_matrix(file=of, m=weights, form=form, var_name=f'layer{i}', quantize=kwargs['quantize'])
        of.write('end')


def quantize_network(input_file: Path, **kwargs):
    quant_file = input_file.parent / (input_file.stem + '.quant.onnx')
    quantize_static(input_file, quant_file, calibration_data_reader=AcasxuDataReader(), quant_format=QuantFormat.QOperator)
    model = onnx.load(str(quant_file))

    # remove `_quantized` suffix from layer names
    suffix = '_quantized'
    for layer in model.graph.initializer:
        if layer.name[-len(suffix):] == suffix:
            layer.name = layer.name[:-len(suffix)]
    return model


def prune_network(model, sparsity, **kwargs):
    pruned_model = []
    for layer in model:
        kth = int(sparsity * layer.size)
        flat_layer = layer.flatten()
        pruned_indices = np.argpartition(np.abs(flat_layer), kth=kth)[:kth]
        pruned_layer = np.array([0 if i in pruned_indices else flat_layer[i] for i in range(layer.size)])
        pruned_model.append(pruned_layer.reshape(layer.shape))
    return pruned_model


def get_weights(model):
    """
    Extract only the weights and biases from the ONNX model of our network
    """
    layers = model.graph.initializer
    res = []

    layer_names = [f'Operation_{i}' for i in range(1, 7)]
    layer_names.append('linear_7')

    weights_str = '_MatMul_W'
    bias_str = '_Add_B'

    for (i, name) in enumerate(layer_names):
        weights = None
        biases = None

        # find weights and biases layers by name
        for layer in layers:
            if layer.name == name + weights_str:
                weights = numpy_helper.to_array(layer)
            if layer.name == name + bias_str:
                biases = numpy_helper.to_array(layer)
        if biases is None or weights is None:
            print(f'Missing data for layer {name}')
        else:
            print(f'layer {name}: weights({weights.shape}) bias({biases.shape})')
            weights = np.insert(weights, 0, biases, axis=0)  # prepend biases to weights matrix
            weights = weights.transpose()
            res.append(weights)
    return res

'''
Add argument parsing:
* -i : input dir
* -o : output dir
* -q : quantization
* -p : pruning
* -s : sparsity percentage after pruning
'''

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Convert to IML and approximate networks from the ACAS Xu benchmark.')
    parser.add_argument('-i', '--input', metavar='i', type=Path, default='./networks_onnx',
                        help='path to input directory', dest='input_dir')
    parser.add_argument('-o', '--output', metavar='o', type=Path, default='./networks_iml',
                        help='path to output directory', dest='output_dir')
    parser.add_argument('-q', '--quantize', action='store_true')
    parser.add_argument('-p', '--prune', action='store_true')
    parser.add_argument('-s', '--sparsity', type=float, help='sparsity rate if pruning', default=0.6)
    args = parser.parse_args(argv)
    return vars(args)

if __name__ == '__main__':
    args_dict = parse_args(sys.argv[1:])
    for file in os.listdir(args_dict['input_dir']):
        print(file)
        if ".quant" in file or ".onnx" not in file or "-opt" in file:
            continue
        input_file = args_dict['input_dir'] / file
        output_file = args_dict['output_dir'] / (input_file.stem + '.iml')
        if args_dict['quantize']:
            onnx_model = quantize_network(input_file, )
        else:
            onnx_model = onnx.load(input_file)
        model_weights = get_weights(onnx_model)
        if args_dict['prune']:
            model_weights = prune_network(model_weights, **args_dict)
        write_network(model_weights, output_file, 'func', **args_dict)
