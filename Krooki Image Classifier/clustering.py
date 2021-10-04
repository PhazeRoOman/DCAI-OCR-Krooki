"""
Created on Wed Sep 15 14:15:28 2021
@author: Hisham Al Hashmi & Anfal Al Yousufi
"""
from sklearn.cluster import KMeans
import preprocessing
import pandas as pd

import utils
import time

"""
    Clustring class scikit-learn's k_means model to cluster the images. The class can call
    on the Preprocessing class to read the files and generate the datafram to be used in the clustering.
    ...
    Attributes
    ----------
        km_model: created using scikit-learn library Kmeans class. currently set to detect 3 clusters
        pp: preprocessing object that reads the image files and creates the dataframe
        df: the unclassified dataframe
        classifed: the classified dataframe with a column called cluster.
    Methods
    -------
    generate_clusters(self)
        this method can be called to read the files and process each image to generate
        a dataframe using the Preprocessing class. Then uses K means to cluster the images.
        
    read_csv(self, path)
        if the df of the images to be classified already exists, then the read_csv 
        method can be used to read the file and cluster the images.
        
   save_classified_df(self, path="output_classified.csv")
        saves the classified dataframe as csv file. if the path is not given, 
        the file is saved as "output_classified.csv" in the same directory as the 
        script.
        
    plot_model_PCA_static(self)/plot_model_3d_interactive(self)
        calls on the respective plotting method in utils to generate plots of 
        the classified model.
    
"""

class Cluster(object):
    
    def __init__(self):
        self.km_model = KMeans(n_clusters=3, random_state=5)
        self.pp = preprocessing.PreProcsses()
        self.df = None
        self.classified = None
    
    def save_classified_df(self, path="_classified.csv"):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.classified.to_csv("./outputs/" + timestr + path)  
        
    def generate_clusters_from_imgs(self):
        model = self.km_model
        self.df = self.pp.generate_df()
        data = self.df[['noise', 'rot_ratio', 'background']]
        model.fit(data)
        data['class'] = model.labels_
        data['image_name'] = self.df.image_name
        self.classified = data
    
    def generate_clusters_from_csv(self):
        model = self.km_model
        self.df = pd.read_csv("./outputs/20210930-074401_unclassified_df.csv")
        data = self.df[['noise', 'rot_ratio', 'background']]
        model.fit(data)
        data['class'] = model.labels_
        data['image_name'] = self.df.image_name
        self.classified = data

    def read_classified_csv(self, path):
        model = self.km_model
        df = pd.read_csv(path)
        self.df = df
        data = df[['noise', 'rot_ratio', 'background']]
        model.fit(data)
        data['class'] = model.labels_
        data['image_name'] = df.image_name
        self.classified = data 
        
        
    def plot_model_PCA_static(self):
        model = self.km_model
        utils.generate_pyplot_plot(self.classified, model.cluster_centers_)
            
        
    def plot_model_3d_interactive(self):
        model = self.km_model
        utils.generate_plotly_plot(self.classified, model.cluster_centers_)
        
