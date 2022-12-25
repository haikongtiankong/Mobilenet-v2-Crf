from histolab.slide import Slide
from histolab.tiler import GridTiler, RandomTiler
from histolab.masks import BinaryMask
import os
import numpy as np

'''
Generating small patches that can be fed into CNN for training, based on the original tif and mask 
of tumor or normal area
PROCESS_PATH: the path of original slide
PROCESSED_PATH: the output path for keeping patches
MASK_PATH_prefix: the root path where we keep all the roi mask of slides'''

slide_name = 'testslide'
PROCESS_PATH = r'F:\医学包1\医学图像处理1\切割\003548-1b\003548-1b - 1.tif'
PROCESSED_PATH = r'D:\self_study\medical_imaging\data'
MASK_PATH_prefix = r'D:\self_study\medical_imaging\mask/'
MASK_PATH = os.path.join(MASK_PATH_prefix, slide_name+'.npy')

test_slide = Slide(path=PROCESS_PATH, processed_path=PROCESSED_PATH)

grid_tiles_extractor = GridTiler(
   tile_size=(768, 768),
   level=0,
   check_tissue=True, # default
   prefix=slide_name+"/", # save tiles in the "grid" subdirectory of slide's processed_path
   suffix=".png", # default
   tissue_percent=20)


#Define the binary mask where we want to extract pacthes
class MyCustomMask(BinaryMask):
   def _mask(self, slide):
      my_mask=np.load(MASK_PATH)
      return my_mask

binary_mask = MyCustomMask()
grid_tiles_extractor.locate_tiles(slide=test_slide, extraction_mask=binary_mask).show()
test_slide.locate_mask(binary_mask).show()
grid_tiles_extractor.extract(slide=test_slide, extraction_mask=binary_mask) #extract the patches included in the binary mask


'''tiles_num = 6
random_tiles_extractor = RandomTiler(
    tile_size=(768, 768),
    n_tiles=tiles_num,
    level=0,
    seed=42,
    check_tissue=True, # default
    tissue_percent=30.0, # default
    prefix=slide_name+"/", # save tiles in the "random" subdirectory of slide's processed_path
    suffix=".png")# default)'''