# Mobilenet-v2-Crf
## Backgrounds
Medical images like H&E (Hematoxylin & eosin) stained Histopathological sections are important references in diagnosing diseases. Doctors' diagnoses may rely more on personal experience and sometimes lead to misdiagnosis. The project presents a pipeline of a deep learning system that includes pre-processing, feature extraction, and post processing of H&E stained WSIs (Whole Slide Images). The model aims at locating positive areas of a certain disease in WSIs. The key point in our model is that the classification result of a CNN (Convolution Neural Network) is optimized by embedding it with a conditional random field. In the field of medical imaging, directly classifying the samples of sections by a CNN may result in inconsistent predictions. The conditional random field considers spatial correlations  among different regions in the tissue slide. According to experimentation, a CNN-CRF model gives an accuracy of 2.5% to 4.9% higher than a CNN model. It is believed that CRF has great potential in the field of medical imaging because many diseases like breast cancer or prostatic cancer all show a spatial correlation between tumor areas in the tissues
## Data
Data that are used during the process is given an example in example file. The dataset used for the project is private.
## Install
Openslide and Histolab are required as the tool to preprocess the original medical images.
They can be found in Install file (coz directly using pip command to install Openslide on Linux system can work but on Windows may fail)
## Pre-processing
### Mask generation
run Mask_gen.py to generate binary mask of annotated tumor or normal area of original digital slide.  
Need the original digital slides in tif formatt and annoted coordinates stored in txt file.  
An example slide and coordinate file are given in example file.  

![Generating Mask](https://github.com/haikongtiankong/Mobilenet-v2-Crf/blob/main/fig/maskgen.png)  
### Patch generation
run Patch_gen.py to generate the small patches that can be fed into a CNN for training.  
The original digital slide and the generated mask are needed.

![Patch generation](https://github.com/haikongtiankong/Mobilenet-v2-Crf/blob/main/fig/patchgen.png)  
### Coordinates generation
run coord_list_gen.py to rename the patches extracted and also acquire a txt file of all patches central coordinates.

### Json file of coordinates
Note that the original txt file that stores coordinates of annotation areas need to be transformed in xml file of a specific format.  
use xml2json.py to transform these xml file in json format.  
An example of xml and json file are given in example file, the format is in line with the json file in Camelyon 17 chanllenges.  
These json file will be used to generate labels of each patch before they are sent to the CNN model.

## Training and post-processing
run train.py to train the model after data pre-processing.  

### Probability maps
run prob_map.py to generate the heatmaps of tested slides, the result is store in npy file.  
run prob_map_show.py to convert npy to tif so that can see the heatmap visually.  

![heatmap](https://github.com/haikongtiankong/Mobilenet-v2-Crf/blob/main/fig/Prob.png)  
