# _*_ coding utf-8 _*_
# 开发人员： RUI
# 开发时间： 2020/9/11
# 文件名称： dataset
# 开发工具： PyCharm

import os
import sys
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


instances_annFile = 'E:/DATA/COCO2014/annotations/instances_val2014.json'
coco = COCO(instances_annFile)

captions_annFile= 'E:/DATA/COCO2014/annotations/captions_val2014.json'
coco_caps = COCO(captions_annFile)

ids = list(coco.anns.keys())



ann_id = np.random.choice(ids)
img_id = coco.anns[ann_id]['image_id']
img = coco.loadImgs(img_id)[0]
url = img['coco_url']

# print URL and visualize corresponding image
print(url)
I = io.imread(url)
plt.figure(figsize=(5,5))
#plt.axis('off')
plt.imshow(I)
plt.show()

# load and display captions
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)





print('end')

