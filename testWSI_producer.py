import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL

np.random.seed(0)
class GridWSIPatchDataset(Dataset):
    """
    Data producer that generate all the square grids, e.g. 3x3, of patches,
    from a WSI and its tissue mask, and their corresponding indices with
    respect to the tissue mask
    """
    def __init__(self, wsi_path, mask_path, image_size=768, patch_size=256,
                 crop_size=224, normalize=True, flip='NONE', rotate='NONE'):
        """
        Initialize the data producer.

        Arguments:
            wsi_path: string, path to WSI file
            mask_path: string, path to mask file in numpy format
            image_size: int, size of the image before splitting into grid, e.g.
                768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
            flip: string, 'NONE' or 'FLIP_LEFT_RIGHT' indicating the flip type
            rotate: string, 'NONE' or 'ROTATE_90' or 'ROTATE_180' or
                'ROTATE_270', indicating the rotate type
        """
        self._wsi_path = wsi_path
        self._mask_path = mask_path
        self._image_size = image_size
        self._patch_size = patch_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._preprocess()

    def _preprocess(self):
        #if self._image_size % self._patch_size != 0:
        #    self._image_size = self._image_size + 2

        self._mask = np.load(self._mask_path)
        self._mask = np.transpose(self._mask).copy()
        #print(len(self._mask[0]))
        self._slide = openslide.open_slide(self._wsi_path)

        X_slide, Y_slide = self._slide.level_dimensions[0]
        X_mask, Y_mask = self._mask.shape
        #X_mask = X_mask + self._image_size
        #Y_mask = Y_mask + self._image_size
        #print(self._mask[17519])

        if X_slide / X_mask != Y_slide / Y_mask:
            raise Exception('Slide/Mask dimension does not match ,'
                            ' X_slide / X_mask : {} / {},'
                            ' Y_slide / Y_mask : {} / {}'
                            .format(X_slide, X_mask, Y_slide, Y_mask))

        #self._resolution = X_slide * 1.0 / X_mask
        #if not np.log2(self._resolution).is_integer():
        #    raise Exception('Resolution (X_slide / X_mask) is not power of 2 :'
        #                    ' {}'.format(self._resolution))

        steps_X = (X_slide - self._image_size)/(self._patch_size/4)#???????????????????????????
        steps_Y = (Y_slide - self._image_size)/(self._patch_size/4)
        self._X_idcs = []
        self._Y_idcs = []
        ix = self._image_size / 2
        iy = self._image_size / 2
        for i in range(int(steps_X+1)):
            iy = self._image_size / 2
            ix = ix + (self._patch_size/4)
            for j in range(int(steps_Y+1)):
                iy = iy + (self._patch_size/4)
                #if not self._mask[int(ix)][int(iy)] == False :
                if (ix < X_slide) and (iy < Y_slide):
                    self._X_idcs.append(ix)
                    self._Y_idcs.append(iy)

        self._idcs_num = len(self._X_idcs)
        print("idcs ", self._idcs_num)

            #raise Exception('Image size / patch size != 0 : {} / {}'.
            #                format(self._image_size, self._patch_size))
        self._patch_per_side = self._image_size // self._patch_size
        self._grid_size = self._patch_per_side * self._patch_per_side

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        x_mask, y_mask = self._X_idcs[idx], self._Y_idcs[idx]

        img = self._slide.read_region(
            (int(x_mask), int(y_mask)), 0, (self._image_size, self._image_size)).convert('RGB')

        # PIL image:   H x W x C
        # torch image: C X H X W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0)/128.0

        # flatten the square grid
        img_flat = np.zeros(
            (self._grid_size, 3, self._crop_size, self._crop_size),
            dtype=np.float32)

        idx = 0
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                # center crop each patch
                x_start = int(
                    (x_idx + 0.5) * self._patch_size - self._crop_size / 2)
                x_end = x_start + self._crop_size
                y_start = int(
                    (y_idx + 0.5) * self._patch_size - self._crop_size / 2)
                y_end = y_start + self._crop_size
                img_flat[idx] = img[:, x_start:x_end, y_start:y_end]

                idx += 1

        return (img_flat, x_mask, y_mask)
