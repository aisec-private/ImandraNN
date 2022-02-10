# CNN library - matrices as functions

## Contents

This folder contains the integer-valued implementation of the Imandra CNN library with matrices implemented as functions (Section 5).

`matrix.iml` contains the implementation of matrices as functions as described in Section 5.

`fully_connected.iml`, `layers.iml` and `top.iml` show the definition of fully-connected layers and their assembly into a feed-forward network.

In addition, `acas_xu_network.iml` shows the definition of a network from the ACAS-Xu benchmark and the verification of one of its properties. The output of this proof is given in `verification_property_1.log`.

`mk_weights_funs.iml` and the `acas_xu` subfolders contain the definitions of the ACAS-Xu networks in the `nnet` format and the scripts used for their conversion to the IML matrices-as-functions form.

The `weights_#*.iml` file contain the values for the weights used in the files described above.

## Usage

```
#redef;;
#use "[top | acas_xu_network].iml";;
```