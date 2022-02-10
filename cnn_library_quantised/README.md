# CNN library - quantised 

## Contents

This folder contains the integer-valued implementation of the Imandra CNN library with matrices implemented as `List`s described in Section 5.

The files mirror those in the folder `cnn_library`:
`fully_connected.iml`, `convolution.iml` and `max_pool.iml` contain the implementations of the respective types of layers. `layers.iml` and `top.iml` show the assembly of the different layer types into a convolutional neural network trained on the "smiley face" dataset.

In addition, `robustness.iml` contains the implementations of the different robustness properties presented in Section 4. The verification of the different definitions of robustness for an example convolutional network trained on the toy "smiley face" dataset is done in `top.iml`.

The `weights_*.iml` file contain the values for the weights used in the files described above.

## Usage

```
#redef;;
#use "top.iml";;
```
