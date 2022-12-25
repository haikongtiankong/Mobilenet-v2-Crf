import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

a=np.load(r'D:\self_study\medical_imaging\prob_map\prob_003548-1b-1.npy')
a = np.uint8(255 * a)
a= cv2.applyColorMap(a, cv2.COLORMAP_JET)
a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
#print(a)
a = np.transpose(a, (1,0,2)).copy()
savepath = r'D:\self_study\medical_imaging\prob_map\prob_003548-1b-1.tif'

plt.imshow(a, cmap='jet', vmin=0, vmax=1)
plt.axis('off')
plt.tight_layout()
plt.colorbar(shrink=0.6)
plt.savefig(savepath, bbox_inches='tight', dpi=600)

#image = Image.fromarray(a)
#image.save('F:/大三下/医学图像处理/data_set/NCRF数据/result文件/test.tif')