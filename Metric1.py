# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:19:42 2020
Metric1 cohen kappa
@author: Alain
"""
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from dataPreproces import load_train_data,load_test_data,load_data
from skimage import io
from sklearn.metrics import cohen_kappa_score


histoImages, masks = load_train_data()

# mcm = multilabel_confusion_matrix(io.imread(masks[0,0]).flatten(),io.imread(masks[0,2]).flatten(),labels=[0,1,2,3,4,5])
# tn = mcm[:, 0, 0]
# tp = mcm[:, 1, 1]
# fn = mcm[:, 1, 0]
# fp = mcm[:, 0, 1]
# tp / (tp + fn)


######segmentation keep
im1 = io.imread(masks[0,0]).flatten()
im2 = io.imread(masks[0,2]).flatten()
# im1[im1>0] = 1
# im2[im2>0] = 1
a = 1
cks =[]
# for i in range(6):
#     for j in range(i+1,6):
mapList1 = []
mapList2 = []
map1 = masks[:,1]
map2 = masks[:,4]
for k in range(50):
    if map1[k]!='NoGT' and map2[k]!='NoGT':
        # mapList1.append(map1[k])
        # mapList2.append(map2[k])
        print("a: ",a)
        print("k: ",k)
        a+=1
        mapList1.extend(io.imread(map1[k]).flatten())
        mapList2.extend(io.imread(map2[k]).flatten())

cks = cohen_kappa_score(mapList1,mapList2,labels=[0,1,2,3,4,5])
print(cks)









