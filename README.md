## Imandra Neural Network Verification Library

### File hierarchy

This repository holds the code for the submission to PPDP2022 "Neural Network Verification with Imandra: a Holistic Approach".
The code is separated into 6 parts:

* `preliminary`: the implementation of a multi-perceptron trained on the Iris Dataset in IML (Section 2)

* `matrix_as_lists/matrix_as_lists_reals`:
  - the library for representing convolutional neural networks (CNN) in IML (Section 3).
  - code for experiments on filter properties (Section 7).

* `matrix_as_lists/matrix_as_lists_integers`:
  - the quantised version of the previous library, necessary to use Imandra's Blast strategy (Section 4).
  - the implementation of multiple definitions of robustness (Section 4), and the evaluation of these multiple definitions.

* `matrix_as_functions`:
  - the formalisation of Neural networks with Matrix as defined as functions as described in Section 5.
  - the verification of a property on a quantised model from the ACAS-Xu benchmark

* `matrix_as_records`:
  - Matrix implemented as records as described in Section ?
  - verification of properties on the ACAS Xu benchmark

* `induction`:
  - the proof of properties on perceptrons by induction reasoning (Section 6).

* `notebooks`: 
  - the python notebooks used to train the network and containing the script to convert trained networks to IML
  - the serialised networks in Keras' serialisation format

### Execution of IML code

The execution of IML code is done via the Imandra CLI. To execute one of the experiments, launch Imandra CLI from the main file's directory and enter the following commands:

```
$ #redef;;
$ #use "entry_file_name.iml";;
```

The functions defined in the main file can then be used in the interactions with the CLI.