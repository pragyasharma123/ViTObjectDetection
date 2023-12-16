import json
import random
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
from collections import defaultdict
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import warnings
warnings.filterwarnings('ignore')
np.float = float    

output_file_path = "/home/ps332/myViT/results.json"
cocoGt = COCO("/home/ps332/myViT/coco_ann2017/annotations/instances_val2017.json")
cocoDt = cocoGt.loadRes(output_file_path)
#cocoDt = cocoGt
imgIds=sorted(cocoGt.getImgIds())
# imgIds=imgIds[0:100]
# imgId = imgIds[np.random.randint(100)]

cocoeval = COCOeval(cocoGt, cocoDt, "bbox")
print("GT \n")
print(cocoeval.cocoGt)
print("DT \n")
print(cocoeval.cocoDt)
cocoeval.params.imgIds  = imgIds
cocoeval.evaluate()
cocoeval.accumulate()
cocoeval.summarize()
