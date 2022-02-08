## Imandra Neural Network Verification Library

### File hierarchy

This repository holds the code for the submission to ITP2022 "Neural Network Verification with Imandra: a Holistic Approach".
The code is separated into 6 parts:

* `preliminary`: the implementation of a multi-perceptron trained on the Iris Dataset in IML (Section 2)

* `cnn_library`:
  - the library for representing convolutional neural networks (CNN) in IML (Section 3). Main file: `top.iml`
  - code for experiments on filter properties (Section 7). Main file: `filter_properties.iml`

* `cnn_library_quantised`:
  - the quantised version of the previous library, necessary to use Imandra's Blast strategy (Section 4). Main file: `top.iml`
  - the implementation of multiple definitions of robustness (Section 4), and the evaluation of these multiple definitions. Main file: `top.iml`
  - the evaluation against an ACAS Xu benchmark model (Section 5). Main file: `acas_xu.iml`

* `cnn_library_matrix_as_functions`:
  - the formalisation of Neural networks with Matrix as defined as functions as described in Section 5.

* `ìnduction`:
  - the proof of properties on perceptrons by induction reasoning (Section 6).

* `notebooks`: 
  - the python notebooks used to train the network and containing the script to convert trained networks to IML
  - the serialised networks in Keras' serialisation format

### Execution of IML code

The execution of IML code is done via the Imandra CLI. To execute one of the experiments, launch Imandra CLI from the main file's directory and enter the following commands:

```
$ #redef;;
$ #use "top_file_name.iml";;
```

The functions defined in the main file can then be used in the interactions with the CLI.
