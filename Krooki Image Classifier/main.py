#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 09:03:39 2021

@author: Hisham Al Hashmi
"""
from clustering import Cluster


if __name__ == "__main__":
    c = Cluster()
    c.generate_clusters_from_imgs()
    c.plot_model_3d_interactive()
    c.plot_model_PCA_static()
