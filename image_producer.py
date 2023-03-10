import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
np.random.seed(0)
from torchvision import transforms  # noqa
from Annotation import Annotation  # noqa


class GridImageDataset(Dataset):
    """
    Data producer that generate a square grid, e.g. 3x3, of patches and their
    corresponding labels from pre-sampled images.
    """
    def __init__(self, data_path, json_path, img_size, patch_size,
                 crop_size=224, normalize=True):
        """
        Initialize the data producer.

        Arguments:
            data_path: string, path to pre-sampled images using patch_gen.py
            json_path: string, path to the annotations in json format
            img_size: int, size of pre-sampled images, e.g. 768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
        """
        self._data_path = data_path
        self._json_path = json_path
        self._img_size = img_size
        self._patch_size = patch_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        self._preprocess()

    def _preprocess(self):
        '''得到一个self._coords的数组，里面包含了每个大patch的[名称，x坐标，y坐标]，如Tumor_024,25417,127565'''
        if self._img_size % self._patch_size != 0:
            self._img_size = self._img_size + 2
            #raise Exception('Image size / patch size != 0 : {} / {}'.
            #                format(self._img_size, self._patch_size))

        self._patch_per_side = self._img_size // self._patch_size
        self._grid_size = self._patch_per_side * self._patch_per_side

        self._pids = list(map(lambda x: x.strip('.json'),
                              os.listdir(self._json_path))) #得到一个list，里面包含了保存json文件下的所有json文件的名字

        self._annotations = {} #用来保存每个tif文件名和其对应annotation的字典
        for pid in self._pids:
            pid_json_path = os.path.join(self._json_path, pid + '.json') #某个单个tif的json文件
            anno = Annotation()
            anno.from_json(pid_json_path) #Initialize the annotation from a json file
            self._annotations[pid] = anno

        self._coords = []
        f = open(os.path.join(self._data_path, 'list.txt'))
        for line in f:
            pid, x_center, y_center = line.strip('\n').split(',')[0:3]
            x_center, y_center = int(x_center), int(y_center)
            self._coords.append((pid, x_center, y_center))
        f.close()
        #print(len(self._coords))

        self._num_image = len(self._coords)

    def __len__(self):
        #print(self._num_image)
        return self._num_image

    def __getitem__(self, idx):
        pid, x_center, y_center = self._coords[idx]

        x_top_left = int(x_center - self._img_size / 2)
        y_top_left = int(y_center - self._img_size / 2)

        # the grid of labels for each patch
        label_grid = np.zeros((self._patch_per_side, self._patch_per_side),
                              dtype=np.float32)
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                # (x, y) is the center of each patch
                x = x_top_left + int((x_idx + 0.5) * self._patch_size)
                y = y_top_left + int((y_idx + 0.5) * self._patch_size)

                #如果坐标在勾画区域内，则小patch的label为1，否则label为0
                if self._annotations[pid].inside_polygons((x, y), True):
                    label = 1
                else:
                    label = 0

                # extracted images from WSI is transposed with respect to
                # the original WSI (x, y)
                label_grid[y_idx, x_idx] = label

        img = Image.open(os.path.join(self._data_path, '{}.png'.format(idx)))
        '''保存patch的路径，比如PATCH_TUMOR_TRAIN中所有图片命名都为类似于 1.png， 且list.txt中的信息顺序和patch相对应'''
        img = img.convert("RGB")

        # color jitter
        img = self._color_jitter(img)

        # use left_right flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label_grid = np.fliplr(label_grid)

        # use rotate
        num_rotate = np.random.randint(0, 4)
        img = img.rotate(90 * num_rotate)
        label_grid = np.rot90(label_grid, num_rotate)

        # PIL image:   H x W x C
        # torch image: C X H X W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0)/128.0

        # flatten the square grid
        img_flat = np.zeros(
            (self._grid_size, 3, self._crop_size, self._crop_size),
            dtype=np.float32)
        label_flat = np.zeros(self._grid_size, dtype=np.float32)

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
                label_flat[idx] = label_grid[x_idx, y_idx]

                idx += 1

        return (img_flat, label_flat)
