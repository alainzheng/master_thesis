
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

    plt.text(-0.11, 0.8, str(nparr.replace('.npy', '').replace('Scores/', '')), fontsize=16)
    plt.text(0.4, 0.8, 'y_true', fontsize=13)
    plt.text(-0.11, 0.2, 'y_pred', fontsize=13)
    
    # if not os.path.exists('Scores'):
    #         os.makedirs('Scores')
    plt.savefig(str(nparr.replace('.npy', ''))+'.png')

def get_scores():

    histoImages, masks = load_train_data()
    
    scores1 = np.empty((6,6))
    scores1[:] = np.NaN
    scores2 = np.empty((6,6))
    scores2[:] = np.NaN 
    scores3 = np.empty((6,6))
    scores3[:] = np.NaN
    
    scores4 = np.empty((6,6))
    scores4[:] = np.NaN 
    scores5 = np.empty((6,6))
    scores5[:] = np.NaN 
    scores6 = np.empty((6,6))
    scores6[:] = np.NaN 
    
    scores7 = np.empty((6,6))
    scores7[:] = np.NaN 
    scores8 = np.empty((6,6))
    scores8[:] = np.NaN 
    scores9 = np.empty((6,6))
    scores9[:] = np.NaN 
    
    for i in range(6):
        for j in range(i+1,6):
            scoreList1 = []
            scoreList2 = []
            scoreList3 = []
            
            scoreList4 = []
            scoreList5 = []
            scoreList6 = []
            
            scoreList7 = []
            scoreList8 = []
            scoreList9 = []
            
            maptrue = masks[:,i] # i is true
            mappred = masks[:,j] # j is predicted
            for k in range(1):
                if maptrue[k]!='NoGT' and mappred[k]!='NoGT':
                    im1 = io.imread(maptrue[k]).flatten()
                    im2 = io.imread(mappred[k]).flatten()
                    lbs = np.unique(im2)
                    """
                    #get binary maps for metrics
                    # th = 0
                    # im1[im1>th] = 1
                    # im2[im2>th] = 1
                    # im1[im1<=th] = 0
                    # im2[im2<=th] = 0
                    """
                    
                    scoreList1.append(f1_score(im1,im2, labels=lbs, average='micro'))
                    scoreList2.append(f1_score(im1,im2, labels=lbs, average='macro'))                    
                    scoreList3.append(f1_score(im1,im2, labels=lbs, average='weighted'))


                    scoreList4.append(jaccard_score(im1,im2, labels=lbs, average='micro'))
                    scoreList5.append(jaccard_score(im1,im2, labels=lbs, average='macro'))                    
                    scoreList6.append(jaccard_score(im1,im2, labels=lbs, average='weighted'))


                    scoreList7.append(recall_score(im1,im2, labels=lbs, average='micro'))
                    scoreList8.append(recall_score(im1,im2, labels=lbs, average='macro'))
                    scoreList9.append(recall_score(im1,im2, labels=lbs, average='weighted'))
                    
            if scoreList1:
                scores1[j,i] = sum(scoreList1)/len(scoreList1)                    
            if scoreList2:
                scores2[j,i] = sum(scoreList2)/len(scoreList2)                
            if scoreList3:
                scores3[j,i] = sum(scoreList3)/len(scoreList3)
                    
            if scoreList4:
                scores4[j,i] = sum(scoreList4)/len(scoreList4)    
            if scoreList5:
                scores5[j,i] = sum(scoreList5)/len(scoreList5)    
            if scoreList6:
                scores6[j,i] = sum(scoreList6)/len(scoreList6)
            
            if scoreList7:
                scores7[j,i] = sum(scoreList7)/len(scoreList7)
            if scoreList8:
                scores8[j,i] = sum(scoreList8)/len(scoreList8)
            if scoreList9:
                scores9[j,i] = sum(scoreList9)/len(scoreList9)
                
    if not os.path.exists('Scores'):
        os.makedirs('Scores')
        
    np.save('Scores/f1_scores_micro.npy', scores1)
    np.save('Scores/f1_scores_macro.npy', scores2)
    np.save('Scores/f1_scores_weighted.npy', scores3)
    
    np.save('Scores/jaccard_scores_micro.npy', scores4)
    np.save('Scores/jaccard_scores_macro.npy', scores5)
    np.save('Scores/jaccard_scores_weighted.npy', scores6)
    
    np.save('Scores/recall_scores_micro.npy', scores7)
    np.save('Scores/recall_scores_macro.npy', scores8)
    np.save('Scores/recall_scores_weighted.npy', scores9)
    
if __name__ == '__main__':
    
    tic = time.perf_counter()
    # nparr = 'Scores/f1_scores_micro.npy'
    # nparr = 'Scores/f1_scores_weighted.npy'
    # nparr = 'Scores/jaccard_scores_micro.npy'
    # nparr = 'Scores/jaccard_scores_weighted.npy'
    # nparr = 'Scores/recall_scores.npy'
    
    # display(nparr)
    
    get_scores()
    
    scores1 = np.load('Scores/f1_scores_micro.npy')
    scores2 = np.load('Scores/f1_scores_macro.npy')
    scores3 = np.load('Scores/f1_scores_weighted.npy')
    
    scores4 = np.load('Scores/jaccard_scores_micro.npy')
    scores5 = np.load('Scores/jaccard_scores_macro.npy')
    scores6 = np.load('Scores/jaccard_scores_weighted.npy')
    
    scores7 = np.load('Scores/recall_scores_micro.npy')
    scores8 = np.load('Scores/recall_scores_macro.npy')
    scores9 = np.load('Scores/recall_scores_weighted.npy')
    
    toc = time.perf_counter()
    print('toc-tic: ' + str(datetime.timedelta(seconds = toc-tic)))
    
