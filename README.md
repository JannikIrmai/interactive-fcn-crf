# interactive-fcn-crf
This repository contains python code that combines a fully convolutional neural network (FCN) with a conditional 
random field (CRF) that can be used to interactively correct flawed predictions.

**UNDER CONSTRUCTION!**

This repository is currently under construction.

# Demo

This example demonstrates the interactive gaussian crf for vertebrae localization in spine CT images on image 
'070' of the VerSe 2019 benchmark dataset [1].
Adjusting the location of one vertebra (L4) adjusts the predicted locations of all other vertebrae in real time.
    

![demo](https://github.com/JannikIrmai/interactive-fcn-crf/blob/main/demo-video.gif)




# References

[1] Anjany Sekuboyina, Malek E Husseini, Amirhossein Bayat, Maximilian Loeffler, Hans Liebl, et al., 
"Verse: A vertebrae labelling and segmentation benchmark for multi-detector ct images," 
MedIA, vol. 73, pp. 102166, 2021.
