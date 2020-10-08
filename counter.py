import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage import io
import os
import numpy as np
import time
from dataPreproces import load_data

if __name__ == '__main__':
    
    print('-'*30)

    print('Run Counter.py...')

    print('-'*30)
    
    tic = time.perf_counter()
    
    histoImages, masks = load_data()
    
    #count the number of slices per slides per expert 
    slideSize = [0, 44, 84, 121, 167, 216, 244]
    slideList = [1, 2, 3, 5, 6, 7]
    
    
    slicePerSlidePerExpert = []
    for i in range(6):#for every expert
        slideCounter = [0, 0, 0, 0, 0, 0, 0]
        for j in range(6):#for every slide
            x = masks[slideSize[j]:slideSize[j+1],i]
            numberNoGT = np.count_nonzero(x == 'NoGT')
            slideCounter[j] = slideSize[j+1]-slideSize[j]-numberNoGT
            
        slideCounter[-1] = sum(slideCounter)
        slicePerSlidePerExpert.append(slideCounter)
    
            
    #count the number of slices per slides per histo images 
    slideCounter = [0, 0, 0, 0, 0, 0, 0]
    for i in range(6): #par slide00
        slide = 'slide00'+str(slideList[i])
        for k in range(len(histoImages)): #par slice
            if slide in histoImages[k]:
                slideCounter[i] +=1
    slideCounter[-1] = sum(slideCounter)
    slicePerSlidePerExpert.append(slideCounter)
    
    
    
    #########################
    
    fig = plt.figure()
    plt.figure(figsize=[23,14.5])
    
    #ax = fig.add_subplot(111)
    col_labels = ['slide001', 'slide002', 'slide003', 'slide005', 'slide006', 'slide007', 'Total']
    row_labels = ['expert1', 'expert2', 'expert3', 'expert4', 'expert5', 'expert6', 'Images']
    colors = [["w","w","w","w","w","w","#1ab3a5"],
              ["w","w","w","w","w","w","#1ab3a5"],
              ["w","w","w","w","w","w","#1ab3a5"],
              ["w","w","w","w","w","w","#1ab3a5"],
              ["w","w","w","w","w","w","#1ab3a5"],
              ["w","w","w","w","w","w","#1ab3a5"],
              ["#1ac3f5","#1ac3f5","#1ac3f5","#1ac3f5","#1ac3f5","#1ac3f5","r"]]
    
    # Draw table
    the_table = plt.table(cellText=slicePerSlidePerExpert,
                          cellColours=colors,
                          colWidths=[0.05] * 7,
                          rowLabels=row_labels,
                          colLabels=col_labels,
                          loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(24)
    the_table.scale(2.3, 8)
    
    # Removing ticks and spines enables you to get the figure only with table
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    plt.tick_params()
    plt.savefig("Figures/ImageCount.png")
    
    
    toc = time.perf_counter()
    print('toc-tic: ' + str(toc-tic))
    print('process time: '+ str(time.process_time()))
    
    
