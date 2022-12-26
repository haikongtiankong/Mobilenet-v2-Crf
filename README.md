# Mobilenet-v2-Crf
## Backgrounds
Medical images like H&E (Hematoxylin & eosin) stained Histopathological sections are important references in diagnosing diseases. Doctors' diagnoses may rely more on personal experience and sometimes lead to misdiagnosis. The project presents a pipeline of a deep learning system that includes pre-processing, feature extraction, and postprocessing of H&E stained WSIs (Whole Slide Images). The model aims at locating positive areas of a certain disease in WSIs. The key point in our model is that the classification result of a CNN (Convolution Neural Network) is optimized by embedding it with a conditional random field. In the field of medical imaging, directly classifying the samples of sections by a CNN may result in inconsistent predictions. The conditional random field considers spatial correlations  among different regions in the tissue slide. According to experimentation, a CNN-CRF model gives an accuracy of 2.5% to 4.9% higher than a CNN model. It is believed that CRF has great potential in the field of medical imaging because many diseases like breast cancer or prostatic cancer all show a spatial correlation between tumor areas in the tissues
## Install
Openslide and Histolab are required as the tool to preprocess the original medical images.
They can be found in Install file (coz directly using pip command to install Openslide on Linux system can work but on Windows may fail)
## Pre-processing
### Mask generation
run Mask_gen.py to generate binary mask of annotated tumor or normal area of original digital slide.  
Need the original digital slides in tif formatt and annoted coordinates stored in txt file.  
An example slide and coordinate file are given in example file.  
![Generating Mask](https://github.com/haikongtiankong/Mobilenet-v2-Crf/blob/main/fig/maskgen.png)
### 
