import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage import io
import os
import numpy as np
import time
from dataPreproces import load_data


def counter_one(expertNb):    
    """
    ############ a table per expert
    expertNb :from 1 to 6
    
    Parameters
    ----------
    expertNb : TYPE
        EN GROS, parco.

    Returns
    -------
    None.

    """
    grades = []
    for sl in range(6):#for every slide 1 2 3 5 6 7
        gradeCounter = []
        gradesPerSlide = [0, 0, 0, 0, 0] #from gleason grade 1 to 5
    
        truthMaps = masks[slideSize[sl]:slideSize[sl+1],expertNb-1]
        
        for tm in truthMaps:
            if tm != 'NoGT':
                im = io.imread(tm)
                gradeCounter.extend(set(im.flatten()))
                            
        for k in range(5):
            gradesPerSlide[k] = np.count_nonzero(np.array(gradeCounter) == k+1)
        grades.append(gradesPerSlide)

    grades = np.array(grades)
    sumGrades = []
    for sg in range(5):
        sumGrades.append(np.sum(grades[:,sg]))
    grades = np.append(grades, [sumGrades],axis = 0)
    np.save('GradeCount/gradesCount' + str(expertNb) + '.npy', grades)
    
    
def counter_two():
    
     ############ a single table, summed grades of experts
    
    grades = []
    for sl in range(6):#for every slide 1 2 3 5 6 7
        gradeCounter = []
        gradesPerSlide = [0, 0, 0, 0, 0] #from gleason grade 1 to 5
        
        for i in range(6):#for every expert
            truthMaps = masks[slideSize[sl]:slideSize[sl+1],i]
            for tm in truthMaps:
                if tm != 'NoGT':
                    im = io.imread(tm)
                    gradeCounter.extend(set(im.flatten()))
                            
        for k in range(5):
            gradesPerSlide[k] = np.count_nonzero(np.array(gradeCounter) == k+1)
        grades.append(gradesPerSlide)
        
    sumGrades = []
    for sg in range(5):
        sumGrades.append(np.sum(grades[:,sg]))
    grades = np.append(grades, [sumGrades],axis = 0)
    np.save('GradeCount/gradesCount.npy', grades)
    

def make_table(expertNumber):
    """
    Parameters
    ----------
    expertNumber : 
        expertNumber from 1 to 6, or can be '' for aggregated experts.

    Returns
    -------
    None.

    """
    
    grades = np.load('GradeCount/gradesCount'+str(expertNumber) +'.npy')
    
    perGrades = [0, 0, 0, 0, 0]
    for gr in range(5):
        perGrades[gr] = (grades[6,gr]/expertSize[expertNumber-1]//0.01)/100 #get result in percentage
    grades = grades.tolist()
    grades.append(perGrades)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    row_labels = ['slide001', 'slide002', 'slide003', 'slide005', 'slide006', 'slide007','Total','Percentage (%)']
    col_labels = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5']
    colors = [["w","w","w","w","w"],
               ["w","w","w","w","w"],
               ["w","w","w","w","w"],
               ["w","w","w","w","w"],
               ["w","w","w","w","w"],
               ["w","w","w","w","w"],
               ["#1ab3a5","#1ab3a5","#1ab3a5","#1ab3a5","#1ab3a5"],
               ["#1ac3f5","#1ac3f5","#1ac3f5","#1ac3f5","#1ac3f5"]]
    
    # Draw table
    the_table = ax.table(cellText=grades,
                          cellColours=colors,
                          colWidths=[0.18] * 5,
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
    plt.text(-0.11, 0.8, 'Expert '+ str(expertNumber), fontsize=13)
    #plt.title('Expert '+ str(expertNumber), fontsize=36)
    plt.savefig('GradeCount/GradeCount'+str(expertNumber)+'.png')
    

if __name__ == '__main__':
    
    t1_start = time.process_time()  

    print('-'*30)

    print('Run counterGrade.py...')

    print('-'*30)
        
    histoImages, masks = load_data()
    
    slideSize = [0, 44, 84, 121, 167, 216, 244]        #slidesize info  
    expertSize = [242, 139, 240, 241, 244, 65]


    ##### no nbr 0 expert, make folder if nonexistant
    # if not os.path.exists('GradeCount'):
    #     os.makedirs('GradeCount')
   
    
    #### handles some problems
    # expertNumber = 2
    # grades = np.load('GradeCount/gradesCount'+str(expertNumber) +'.npy')
    # sumGrades = []
    # for sg in range(5):
    #     sumGrades.append(np.sum(grades[:,sg]))
    # grades = np.append(grades, [sumGrades],axis = 0)    
    # np.save('GradeCount/gradesCount' + str(expertNumber) + '.npy', grades)

   
    #### not tested yet, aggregated experts
    # counter_two()
    # make_table('')
    
    
    
    ##### for every expert
    for exp in range(1,7):
        counter_one(exp)
        make_table(exp)
        
        
    ### test to put every table together, doesn't work
    # plt.figure()
    # plt.figure(figsize=[14,10])
    # for i in range(1,7):
    #     im = io.imread('GradeCount/GradeCount'+str(i)+'.png')
    #     plt.subplot(2,3,i)
    #     io.imshow(im) 
    
    t1_stop = time.process_time() 
    print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)     
    
