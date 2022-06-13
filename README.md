## CheckInn: Imandra Neural Network Verification Library

### File hierarchy

This repository holds the code for the submission to PPDP2022 "CheckINN: Wide Range Neural Network Verification in Imandra".
The code is separated into multiple parts:

* `matrix_as_lists/matrix_as_lists_reals`:
  - Implementation of convolutional neural networks (CNN) in IML (Section 3).
  - `extreme_value_lemma.iml`: lemma 4.1 for proving structural properties of CNNs (Section 4.2)

* `matrix_as_lists/matrix_as_lists_integers`:
  - Quantised version of the previous implementation, necessary to use Imandra's `blast` strategy (Section 5).
  - Implementation of multiple robustness definitions (Section 5), and their evaluation.

* `matrix_as_functions/matrix_as_functions_integers`:
  - Implementation of NN with matrices as functions (Section 6.1).
  - Verification of properties from the ACAS Xu benchmark on pruned networks.

* `matrix_as_functions/matrix_as_functions_reals`:
  - Implementation of NN with matrices as functions with support for real numbers (Section 6.2, appendix E.1).
  - Verification of properties from the ACAS Xu benchmark on pruned networks.

* `matrix_as_records`:
  - Implementation of NN with matrices as records (Appendix E.2).
  - Verification of properties from the ACAS Xu benchmark on pruned networks.

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
