
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:19:42 2020
Metric1 recall, f1score and jaccard
@author: Alain
"""
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from dataPreproces import load_train_data,load_test_data,load_data
from skimage import io
from sklearn.metrics import recall_score, f1_score, jaccard_score, cohen_kappa_score
import matplotlib.pyplot as plt
import time
import datetime

import os

def display(nparr):
    
    scores = np.load(nparr)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    row_labels = ['Expert 1', 'Expert 2', 'Expert 3', 'Expert 4', 'Expert 5', 'Expert 6']
    col_labels = ['Expert 1', 'Expert 2', 'Expert 3', 'Expert 4', 'Expert 5', 'Expert 6']
    # #colors = [["w","w","w","w","w"],
    #            ["w","w","w","w","w"],
    #            ["w","w","w","w","w"],
    #            ["w","w","w","w","w"],
    #            ["w","w","w","w","w"],
    #            ["w","w","w","w","w"],
    #            ["#1ab3a5","#1ab3a5","#1ab3a5","#1ab3a5","#1ab3a5"],
    #            ["#1ac3f5","#1ac3f5","#1ac3f5","#1ac3f5","#1ac3f5"]]
    
    # Draw table
    the_table = ax.table(cellText=scores,
                          #cellColours=colors,
                          colWidths=[0.18] * scores.shape[0],
                          rowLabels=row_labels,
                          colLabels=col_labels,
                          edges='closed',
                          loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(0.8, 1.2)
    ax.axis('off')    

    # Removing ticks and spines enables you to get the figure only with table
    #plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    #plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    #plt.tick_params()
    plt.text(-0.11, 0.8, str(nparr.replace('.npy', '')), fontsize=13)
    #plt.title('Expert '+ str(expertNumber), fontsize=36)
    if not os.path.exists('Scores'):
            os.makedirs('Scores')
    plt.savefig('Scores/'+str(nparr.replace('.npy', ''))+'.png')

def get_score():


    histoImages, masks = load_train_data()
    
    scores1 = np.empty((6,6))
    scores1[:] = np.NaN
    scores2 = np.empty((6,6))
    scores2[:] = np.NaN 
    scores3 = np.empty((6,6))
    scores3[:] = np.NaN
    scores4 = np.empty((6,6))
    scores4[:] = np.NaN 
    for i in range(6):
        for j in range(i+1,6):
            scoreList1 = []
            scoreList2 = []
            scoreList3 = []
            scoreList4 = []
            maptrue = masks[:,i]
            mappred = masks[:,j]
            for k in range(167):
                if maptrue[k]!='NoGT' and mappred[k]!='NoGT':
                    im1 = io.imread(maptrue[k])
                    im2 = io.imread(mappred[k])
                    #get binary maps for metrics
                    # th = 0
                    # im1[im1>th] = 1
                    # im2[im2>th] = 1
                    # im1[im1<=th] = 0
                    # im2[im2<=th] = 0
                    scoreList1.append(f1_score(im1.flatten(),im2.flatten(), labels=np.unique(im2), average='micro'))
                    scoreList2.append(jaccard_score(im1.flatten(),im2.flatten(), labels=np.unique(im2), average='micro'))
                    
                    scoreList3.append(f1_score(im1.flatten(),im2.flatten(), labels=np.unique(im2), average='weighted'))
                    scoreList4.append(jaccard_score(im1.flatten(),im2.flatten(), labels=np.unique(im2), average='weighted'))
            if scoreList1:
                scores1[j,i] = sum(scoreList1)/len(scoreList1)
                    
            if scoreList2:
                scores2[j,i] = sum(scoreList2)/len(scoreList2)
                
            if scoreList3:
                scores3[j,i] = sum(scoreList3)/len(scoreList3)
                    
            if scoreList4:
                scores4[j,i] = sum(scoreList4)/len(scoreList4)
    
    np.save('f1_scores_micro.npy', scores1)
    np.save('jaccard_scores_micro.npy', scores2)
    
    np.save('f1_scores_weighted.npy', scores3)
    np.save('jaccard_scores_weighted.npy', scores4)
    
if __name__ == '__main__':
    
    tic = time.perf_counter()
    nparr = 'jaccard_scores_micro.npy'
    
    scores = np.load(nparr)

    for i in range(6):
        for j in range(i+1,6):
            scores[j,i] = round(scores[j,i],3) 
        
    np.save(nparr,scores)
    display('jaccard_scores_micro.npy')
    toc = time.perf_counter()
    print('toc-tic: ' + str(datetime.timedelta(seconds = toc-tic)))
    
