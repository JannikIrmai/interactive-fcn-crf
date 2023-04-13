# Interactive FCN-CRF
This repository contains the code corresponding to the paper 
"Pushing the limits of an FCN and a CRF towards near-ideal vertebrae labelling" [1].
In particular, it includes the fully convolutional neural network "HRNet3D", implemented in pytorch, as 
well as an implementation of a gaussian conditional random field (CRF) with an interactive user interface.


# Demo

This example demonstrates the interactive gaussian CRF for vertebrae localization in spine CT images on image 
'070' of the VerSe 2019 benchmark dataset [2].
Adjusting the location of one vertebra (L4) adjusts the predicted locations of all other vertebrae in real time.

![demo](https://github.com/JannikIrmai/interactive-fcn-crf/blob/main/demo-video.gif)



# Usage

To install the interactive gaussian CRF module, clone this repository and execute

```
pip install interactive-gaussian-crf/
```

The usage of the crf module is demonstrated in two examples in the examples directory:
```
cd examples
python toy_crf_example.py
python spine_crf_example.py
```
More examples can be found in ``interactive_gaussian_crf/tests/test_gaussian_crf.py`` file.


Once the matplotlib window is created the following interactive operations are supported:

- Adjust the location of one variable by drag and drop. 
The locations of all other variables will be updated in real time. 
All variables whose location was adjusted are highlighted by a black circle.
The adjustment can be undone by clicking on the variable.
- Press ``u`` to show the unary distributions of all variables.
- Press ``e`` to (de)activate the display of the covariance ellipses around the MAP locations.
- Press ``m`` to (de)activate the display of the MAP locations.

# HRNet3D

The pytorch implementation of the HRNet3D is in the hr_net_3d directory.
How to use the HRNet3D for vertebrae localization in CT images of the human spine is demonstrated in 
the vertebrae_labeling directory.
The used datasets are publicly available here: https://github.com/anjany/verse.

# References

[1] Anjany Sekuboyina, Jannik Irmai, Bjoern Andres, Suprosanna Shit, Bjoern Menze, 
"Pushing the limits of an FCN and a CRF towards near-ideal vertebrae labelling,"
IEEE International Symposium on Biomedical Imaging (ISBI), 2023.

[2] Anjany Sekuboyina, Malek E Husseini, Amirhossein Bayat, Maximilian Loeffler, Hans Liebl, et al., 
"Verse: A vertebrae labelling and segmentation benchmark for multi-detector ct images," 
MedIA, vol. 73, pp. 102-166, 2021.
