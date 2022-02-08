from nnet import NNet
import numpy as np
import math

def write_matrix(file, m):
  file.write('[\n')
  for line in m:
    file.write('  [')
    for x in line:
      file.write('{:.5f}; '.format(x))
    file.write(']; \n')
  file.write(']; \n')

def nnet_to_iml_weights():
  nnet_file = "./acas_xu.nnet"
  iml_file = "./acas_xu.iml"

  n = NNet(nnet_file)
  file = open(iml_file, 'w')

  for w, b, i in zip(n.weights, n.biases, range(len(n.weights))):
    w = np.insert(w, 0, b, axis=1)
    file.write(f'\nlet layer{i} = ')
    write_matrix(file, w)
  
  file.close()

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return int(math.floor(n*multiplier + 0.5) / multiplier)


def write_matrix_quantized(file, m):
  file.write('[\n')
  for line in m:
    file.write('  [')
    for x in line:
      file.write('{:d}; '.format(round_half_up(x)))
    file.write(']; \n')
  file.write('] \n')

def nnet_to_iml_weights_quantized():
  nnet_file = "./acas_xu.nnet"
  iml_file = "./acas_xu.iml"

  n = NNet(nnet_file)
  file = open(iml_file, 'w')

  file.write('module Weights = struct\n')
  for w, b, i in zip(n.weights, n.biases, range(len(n.weights))):
    w = np.insert(w, 0, b, axis=1)
    file.write(f'\nlet layer{i} = ')
    write_matrix_quantized(file, w)
  
  file.write('end')
  file.close()

nnet_to_iml_weights_quantized()
