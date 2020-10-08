import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage import io
import os
import numpy as np
import time


def create_images(path):
    slices = [(path + '/' + s) for s in os.listdir(path)]
    return slices


def create_train_data():
    dataDirList = os.listdir('Data/')
    histoImages = create_images('Data/' + dataDirList[-1])[:167] # create list of HE images
    
    groundTruths = []  # create a set of list of maps, groundTruths[dataDirList][imageNumber]
    for h in range(len(dataDirList)-1):
        groundTruths.append(create_images('Data/' + dataDirList[h]))
        
    masks = []
    for singleHistoImage in histoImages: #for every image in Train Imgs
        mask = ['NoGT', 'NoGT', 'NoGT', 'NoGT', 'NoGT', 'NoGT']
        for expertNumber in range(6): # for every expert
            # sliceName = singleHistoImage.replace('.jpg', '_classimg_nonconvex.png').replace('Train Imgs', dataDirList[j]) #ex: slide006_core077_classimg_nonconvex.png
            sliceName = singleHistoImage.replace('.jpg', '').replace('Data/Train Imgs/', '') #ex: slide006_core077
            # print(sliceName)
            for gtNumber in range(len(groundTruths[expertNumber])):
                if sliceName in groundTruths[expertNumber][gtNumber]:
                    mask[expertNumber] = groundTruths[expertNumber][gtNumber]
        masks.append(mask)
    
    np.save('Data/trainImages.npy', histoImages)
    np.save('Data/trainMasks.npy', masks)
    
def create_test_data():
    
    dataDirList = os.listdir('Data/')
    histoImages = create_images('Data/' + dataDirList[-1])[167:] # create list of HE images
    
    groundTruths = []  # create a set of list of maps, groundTruths[dataDirList][imageNumber]
    for h in range(len(dataDirList)-1):
        groundTruths.append(create_images('Data/' + dataDirList[h]))
        
    masks = []
    for singleHistoImage in histoImages: #for every image in Train Imgs
        mask = ['NoGT', 'NoGT', 'NoGT', 'NoGT', 'NoGT', 'NoGT']
        for expertNumber in range(6): # for every expert
            # sliceName = singleHistoImage.replace('.jpg', '_classimg_nonconvex.png').replace('Train Imgs', dataDirList[j]) #ex: slide006_core077_classimg_nonconvex.png
            sliceName = singleHistoImage.replace('.jpg', '').replace('Data/Train Imgs/', '') #ex: slide006_core077
            # print(sliceName)
            for gtNumber in range(len(groundTruths[expertNumber])):
                if sliceName in groundTruths[expertNumber][gtNumber]:
                    mask[expertNumber] = groundTruths[expertNumber][gtNumber]
        masks.append(mask)
    
    np.save('Data/testImages.npy', histoImages)
    np.save('Data/testMasks.npy', masks)
    


def load_train_data():

    imgs = np.load('Data/trainImages.npy')

    masks = np.load('Data/trainMasks.npy')

    return imgs, masks

def load_test_data():

    imgs = np.load('Data/testImages.npy')

    masks = np.load('Data/testMasks.npy')

    return imgs, masks


def load_data():

    imgs = np.load('Data/images.npy')

    masks = np.load('Data/masks.npy')

    return imgs, masks


if __name__ == '__main__':
    
    print('-'*30)

    print('Data creation...')

    print('-'*30)

    create_train_data()
    create_test_data()
        
    #imgs, masks = load_data()
    
    
    
    
    
    
    
    