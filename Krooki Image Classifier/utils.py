"""
Created on Wed Sep 15 14:27:06 2021
@author: Hisham Al Hashmi & Anfal Al Yousufi
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import time 
from sklearn.decomposition import PCA

pio.renderers.default='browser'


'''
    This script is called from clustering.py. It can be used to generate plots of the 
    model. The 3d plot uses plotly while in the case of the 2d plot, we are using 
    PCA from scikit learn to first reduce the dimensions to 2 and then plot using 
    matplotlib.pyplot.
    ...
    Methods
    -------
    generate_plotly_plot(classifed_df, centers)        
        takes in the df of the calssifed values and the cluster centers and
        plots a 3d plot on your default browser.
        
        
    generate_pyplot_plot(classifed_df, centers)
        takes in the original df as well as the cluster centres, and a list of 
        the classifcation for each point and plots a 2d plot after using PCA to reduce the 
        dimensionality.
'''


def generate_plotly_plot(classifed_df, centers):    
    fig = px.scatter_3d(classifed_df, x='noise', z='rot_ratio', y='background',
              color='class', hover_name='image_name', labels={
                     "noise": "Noise Score (Scaled)",
                     "background": "Background Score (Scaled)",
                     "rot_ratio": "Rotation Score (Scaled)"
                 },
                title="Krookie Images Clustering Model")
    fig.update_layout(scene={'xaxis' : {'visible': True, 'showticklabels': False},
                            'yaxis' : {'visible': True, 'showticklabels': False},
                            'zaxis' : {'visible': True, 'showticklabels': False}})
    fig.show()

    

def generate_pyplot_plot(classifed_df, centers):
    pca_model = PCA(2)
    data = classifed_df[['noise', 'rot_ratio', 'background']]
    labels = classifed_df['class']
    pca_model.fit(data)
    Z = np.array(pca_model.transform(data))
    pca_df = pd.DataFrame(data=Z, columns=['PCA_1', 'PCA_2'])
    pca_df['class'] = labels
       
    class_0 = pca_df[pca_df['class'] == 0]
    class_1 = pca_df[pca_df['class'] == 1]
    class_2 = pca_df[pca_df['class'] == 2]

    plt.scatter(class_0['PCA_1'], class_0['PCA_2'], c='olivedrab',
                label='Cluster 0')
    plt.scatter(class_1['PCA_1'], class_1['PCA_2'], c='royalblue',
                label='Cluster 1')
    plt.scatter(class_2['PCA_1'], class_2['PCA_2'], c='darkorange',
                label='Cluster 2')
    plt.scatter(pca_model.transform(centers)[:, 0],
                pca_model.transform(centers)[:, 1],
                marker="*", c='black', label="Cluster centers")
    plt.ylabel("PCA_2")
    plt.xlabel("PCA_1")
    plt.title("PCA plot of images and cluster centres")
    plt.legend()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig("./outputs/" + timestr +  "_2D plot.pdf")
    plt.savefig("./outputs/" + timestr + "_2D plot.png")
    plt.show()
