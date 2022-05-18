# CNN library

## Contents

This folder contains the real-valued implementation of the Imandra CNN library with matrices implemented as `List`s (Section 3).

`convolution.iml` and `max_pool.iml` contain the implementations of the respective types of layers as presented in Sections 3.1 and 3.2.

`layers.iml` and `top.iml` show the assembly of the different layer types into a convoltuional neural network (Section 3.3) for a classifier for the toy "smiley face" dataset.

`filter_properties.iml` presents the explainability experiments described in Section 7. 

The `weights_*.iml` file contain the values for the weights used in the files described above.


## Usage

```
#redef;;
#use "[top | filter_properties].iml";;
```