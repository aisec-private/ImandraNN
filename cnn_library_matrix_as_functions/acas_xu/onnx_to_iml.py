import os

import numpy as np
import onnx
from numpy import random as rand
from onnx import numpy_helper
from onnxruntime.quantization import quantize_static, QuantType
from onnxruntime.quantization.calibrate import CalibrationDataReader

from pathlib import Path

rand.seed(42)


class AcasxuDataReader(CalibrationDataReader):
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


def write_matrix_quantized(file, m):
    file.write('[\n')
    for line in m:
        file.write('  [ ')
        for x in line:
            file.write(f'{x}; ')
        file.write(' ];\n')
    file.write(']\n')


def onnx_to_iml_quantized(src_file):
    model_fp32 = Path(src_file)
    model_quant = model_fp32.parent / (model_fp32.stem + ".quant.onnx")

    # model quantization using onnxruntime
    quantize_static(model_fp32, model_quant, calibration_data_reader=AcasxuDataReader(), weight_type=QuantType.QUInt8)

    model = onnx.load(str(model_quant))
    layers = model.graph.initializer

    layer_names = [f'Operation_{i}' for i in range(1, 7)]
    layer_names.append('linear_7')

    # write to iml file
    iml_file = dst_dir / (src_file.stem + '.iml')
    dst_file = Path(iml_file)
    of = open(dst_file, 'w')
    of.write('module Weights = struct\n')

    for (i, name) in enumerate(layer_names):
        weights = None
        biases = None

        # find weights and biases layers by name
        for layer in layers:
            if layer.name == name + '_MatMul_W_quantized':
                weights = numpy_helper.to_array(layer)
            if layer.name == name + '_Add_B_quantized':
                biases = numpy_helper.to_array(layer)
        if biases is None or weights is None:
            print(f'Missing data for layer {name}')

        else:
            print(f'layer {name}: weights({weights.shape}) bias({biases.shape})')
            weights = np.insert(weights, 0, biases, axis=0)  # prepend biases to weights matrix
            weights = weights.transpose()
            of.write(f'\nlet layer{i} = ')
            write_matrix_quantized(of, weights)

    of.write('end')
    of.close()
    return iml_file


    return model_quant


if __name__ == '__main__':
    src_dir = Path('./networks_onnx')
    dst_dir = Path('./networks_iml')
    for file in os.listdir(src_dir):
        print(file)
        if ".quant" in file or ".onnx" not in file or "-opt" in file:
            continue
        src_path = Path(src_dir) / file
        out_path = onnx_to_iml_quantized(src_path)
        print(f'Wrote iml to {out_path}.')
