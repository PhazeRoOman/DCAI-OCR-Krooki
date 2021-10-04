"""
Created on Wed Sep 15 13:35:45 2021
@author: Hisham Al Hashmi & Anfal Al Yousufi
"""
import pandas as pd
from skimage.restoration import estimate_sigma
import os 
import numpy as np
import cv2
import random as r
import time 



"""
    Clustring class scikit-learn's k_means model to cluster the images. The class can call
    on the Preprocessing class to read the files and generate the datafram to be used in the clustering.
    ...
    Attributes
    ----------
        df: the df generated 
        folder_path: the path of where the images are stored.
        normilized_df: the df after it is rescaled and normilized
    Methods
    -------
    generate_df(self)
        genreates a df that can then be used for clustring. this df is already normilized
    
    save_df(self, path="unclassed_df.csv")
        saves the generated df to a csv. if the path is not given, the file will be saved as 
        unclassed_df.csv in the same directory as the script
        
    get_images(self)
        gets a list of the image names available from the given direcotry
    
    get_image_info(self, image_name)
        gets the background, noise and rot_ratio scores of the image and returns 
        a dictonary that gets appended to the df
    
    get_noise(self, image)
        Using Scikit-image to get the gausian noise of an image. 
    
    noise_scale(self)
        standerdizes the noise scores by centering them around 0
        
    get_rot_ratio(self, image)
        gets the ratio of height to width of each image.
    
    rot_rescale(self)
        takes in the rotation ratio values for all images and augments the column
        to generate a bimodal distibution
        
    get_background(self, image)
        calculates a score for the bacground color of the document by sampling 100 random pixels from
        the image.
        
    background_scale(self)
        standerdizes the background scores by centering them around 0
    """
class PreProcsses(object):
    def __init__(self):
        self.df = pd.DataFrame(columns=['image_name', 'noise', 'rot_ratio', 
                                'background'])
        self.folder_path = './data/imgs/'
        self.normilized_df = None
        
        
    def save_df(self, path="_unclassified_df.csv"):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.df.to_csv("./outputs/" + timestr + path)
    
    
    def get_images(self):
        list_of_images = os.listdir(self.folder_path)
        filtered = filter(lambda x: x.split(".")[-1] in ["png", "PNG", 
                                                             "JPG", "jpg", 
                                                             "JPEG", "jpeg"]
                          , list_of_images)
        return list(filtered)


    def generate_df(self):
        for image in self.get_images():
            self.df = self.df.append(self.get_image_info(image), 
                                      ignore_index=True)        
        return self.normilize_df()
    

           
    
    def get_rot_ratio(self, image):
        (H, W, _) = image.shape
        return H/W
    
    
    def rot_scale(self):
        s = self.df.rot_ratio
        std = s.std()
        mean = s.mean()
        left = s[abs(s-mean)>std].copy()
        right = s[abs(s-mean)<=std].copy()
        left_arr = (left-left.min())/(left.max()-left.min())
        right_arr = (right+right.min())/(right.max()-right.min())
        merged = left_arr.append(right_arr)
        merged.sort_index(inplace=True)
        merged = (merged - merged.min())/(merged.max() - merged.min())
        return merged
    
    
    def get_background(self, image):
        (H, W, _) = image.shape
        loPixels = [] 
        # the number of elements in the final list of pixels
        for i in range(100): 
            # random choice of pixels chosen and the color is calculated
            pixel = image[r.randint(0,H-1), r.randint(0,W-1)].sum() 
            while pixel < 60: # if the color is black, pick a new point
                pixel = image[r.randint(0,H-1), r.randint(0,W-1)].sum()
            loPixels.append(pixel)
        color = 764 - np.mean(loPixels)
        var = np.var(loPixels)
        return var * color 
    

    def background_scale(self):
        s = self.df.background
        scaled = (s-s.min())/(s.max() - s.min())
        return scaled
    
    
    
    def get_noise(self, image):
        return estimate_sigma(image, multichannel=True, 
                              average_sigmas=True)
    
    def noise_scale(self):
        s = self.df.noise
        scaled = (s-s.min())/(s.max() - s.min())
        return scaled
    
    
    
    def get_image_info(self, image_name):
        image_path = os.path.join(self.folder_path, image_name)
        image = cv2.imread(image_path)
        noise = self.get_noise(image)
        rot_ratio = self.get_rot_ratio(image)
        background = self.get_background(image)
        return {'image_name': image_name, 'noise': noise, 
                'rot_ratio': rot_ratio, 'background' : background}

    
    
    def normilize_df(self):
        df = self.df
        names = df.image_name
        df.rot_ratio = self.rot_scale()
        df.background = self.background_scale()
        df.noise = self.noise_scale()
        normilized_df =(df-df.mean())/df.std()
        normilized_df.image_name = names
        self.normilized_df = normilized_df
        return normilized_df