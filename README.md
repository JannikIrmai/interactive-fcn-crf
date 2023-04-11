# interactive-fcn-crf
This repository contains python code that combines a fully convolutional neural network (FCN) with a conditional 
random field (CRF) that can be used to interactively correct flawed predictions.

**UNDER CONSTRUCTION!**

This repository is currently under construction.

# Demo

This example demonstrates the interactive gaussian CRF for vertebrae localization in spine CT images on image 
'070' of the VerSe 2019 benchmark dataset [1].
Adjusting the location of one vertebra (L4) adjusts the predicted locations of all other vertebrae in real time.

![demo](https://github.com/JannikIrmai/interactive-fcn-crf/blob/main/demo-video.gif)



# Usage

To install the interactive gaussian crf module, clone this repository and execute

```
pip install interactive-gaussian-crf/
```

The usage of the crf module is demonstrated in two examples in the examples directory:
```
cd examples
python toy_example.py
python spine_example.py
```

Once the matplotlib window is created the following interactive operations are supported:

- Adjust the location of one variable by drag and drop. 
The locations of all other variables will be updated in real time. 
All variables whose location was adjusted are highlighted by a black circle.
The adjustment can be undone by clicking on the variable.
- Press ``u`` to show the unary distributions of all variables.
- Press ``e`` to (de)activate the display of the covariance ellipses around the MAP locations.
- Press ``m`` to (de)activate the display of the MAP locations.


# References

[1] Anjany Sekuboyina, Malek E Husseini, Amirhossein Bayat, Maximilian Loeffler, Hans Liebl, et al., 
"Verse: A vertebrae labelling and segmentation benchmark for multi-detector ct images," 
MedIA, vol. 73, pp. 102166, 2021.
