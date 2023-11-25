# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:57:33 2023

@author: 532807
"""

import tkinter as tk
from tkinter import ttk

variables = dict()
models1 = dict()
models2 = dict()

root = tk.Tk()
root.title('Ai Toolbox - Machine Learning - Time Series Analysis')
root.columnconfigure(0, weight=1)

mc = ttk.Frame(root)
mc.grid(padx=10, pady=10, sticky=(tk.W + tk.E))
mc.columnconfigure(0, weight=1)



def selectAllClustering():
    for i in models1.keys():
        models1[i].set(1)


def unselectAllClustering():
    for i in models1.keys():
        models1[i].set(0)



clustering_info = ttk.LabelFrame(mc, text='Clustering')
clustering_info.grid(padx=5, pady=5, sticky=(tk.W + tk.E))
for i in range(4):
    clustering_info.columnconfigure(i, weight=1)
    
    
models1["clustering_kmeans"] = tk.BooleanVar()
clustering_kmeans = ttk.Checkbutton(clustering_info, text="K-Means Clustering (kmeans)", variable=models1["clustering_kmeans"], onvalue = 1, offvalue = 0)
clustering_kmeans.grid(row=0, column=0, sticky=(tk.W + tk.E))

models1["clustering_ap"] = tk.BooleanVar()
clustering_ap = ttk.Checkbutton(clustering_info, text="Affinity Propagation (ap)", variable=models1["clustering_ap"], onvalue = 1, offvalue = 0)
clustering_ap.grid(row=0, column=1, sticky=(tk.W + tk.E))

models1["clustering_meanshift"] = tk.BooleanVar()
clustering_meanshift = ttk.Checkbutton(clustering_info, text="Mean Shift Clustering (meanshift)", variable=models1["clustering_meanshift"], onvalue = 1, offvalue = 0)
clustering_meanshift.grid(row=0, column=2, sticky=(tk.W + tk.E))

models1["clustering_sc"] = tk.BooleanVar()
clustering_sc = ttk.Checkbutton(clustering_info, text="Spectral Clustering (sc)", variable=models1["clustering_sc"], onvalue = 1, offvalue = 0)
clustering_sc.grid(row=0, column=3, sticky=(tk.W + tk.E))



models1["clustering_hclust"] = tk.BooleanVar()
clustering_hclust = ttk.Checkbutton(clustering_info, text="Agglomerative Clustering (hclust)", variable=models1["clustering_hclust"], onvalue = 1, offvalue = 0)
clustering_hclust.grid(row=1, column=0, sticky=(tk.W + tk.E))

models1["clustering_dbscan"] = tk.BooleanVar()
clustering_dbscan = ttk.Checkbutton(clustering_info, text="Density-Based Spatial Clustering (dbscan)", variable=models1["clustering_dbscan"], onvalue = 1, offvalue = 0)
clustering_dbscan.grid(row=1, column=1, sticky=(tk.W + tk.E))

models1["clustering_optics"] = tk.BooleanVar()
clustering_optics = ttk.Checkbutton(clustering_info, text="OPTICS Clustering (optics)", variable=models1["clustering_optics"], onvalue = 1, offvalue = 0)
clustering_optics.grid(row=1, column=2, sticky=(tk.W + tk.E))

models1["clustering_birch"] = tk.BooleanVar()
clustering_birch = ttk.Checkbutton(clustering_info, text="Birch Clustering (birch)", variable=models1["clustering_birch"], onvalue = 1, offvalue = 0)
clustering_birch.grid(row=1, column=3, sticky=(tk.W + tk.E))



models1["clustering_kmodes"] = tk.BooleanVar()
clustering_kmodes = ttk.Checkbutton(clustering_info, text="K-Modes Clustering (kmodes)", variable=models1["clustering_kmodes"], onvalue = 1, offvalue = 0)
clustering_kmodes.grid(row=2, column=0, sticky=(tk.W + tk.E))



ttk.Button(clustering_info, text="Select All", command=selectAllClustering).grid(row=8, column=2, padx=5, pady=5, sticky=(tk.W + tk.E))
ttk.Button(clustering_info, text="Unselect All", command=unselectAllClustering).grid(row=8, column=3, padx=5, pady=5, sticky=(tk.W + tk.E))



def selectAllAD():
    for i in models2.keys():
        models2[i].set(1)


def unselectAllAD():
    for i in models2.keys():
        models2[i].set(0)



ad_info = ttk.LabelFrame(mc, text='Clustering')
ad_info.grid(padx=5, pady=5, sticky=(tk.W + tk.E))
for i in range(4):
    ad_info.columnconfigure(i, weight=1)


models2["ap_abod"] = tk.BooleanVar()
ad_abod = ttk.Checkbutton(ad_info, text="Angle-base Outlier Detection (abod)", variable=models2["ap_abod"], onvalue = 1, offvalue = 0)
ad_abod.grid(row=0, column=0, sticky=(tk.W + tk.E))

models2["ad_cluster"] = tk.BooleanVar()
ad_cluster= ttk.Checkbutton(ad_info, text="Clustering-Based Local Outlier (cluster)", variable=models2["ad_cluster"], onvalue = 1, offvalue = 0)
ad_cluster.grid(row=0, column=1, sticky=(tk.W + tk.E))

models2["ad_cof"] = tk.BooleanVar()
ad_cof = ttk.Checkbutton(ad_info, text="Connectivity-Based Local Outlier (cof)", variable=models2["ad_cof"], onvalue = 1, offvalue = 0)
ad_cof.grid(row=0, column=2, sticky=(tk.W + tk.E))

models2["ad_iforest"] = tk.BooleanVar()
ad_iforest = ttk.Checkbutton(ad_info, text="Isolation Forest (iforest)", variable=models2["ad_iforest"], onvalue = 1, offvalue = 0)
ad_iforest.grid(row=0, column=3, sticky=(tk.W + tk.E))



models2["ad_histogram"] = tk.BooleanVar()
ad_histogram = ttk.Checkbutton(ad_info, text="Histogram-based Outlier Detection (histogram)", variable=models2["ad_histogram"], onvalue = 1, offvalue = 0)
ad_histogram.grid(row=1, column=0, sticky=(tk.W + tk.E))

models2["ad_knn"] = tk.BooleanVar()
ad_knn = ttk.Checkbutton(ad_info, text="K-Nearest Neighbors Detector (knn)", variable=models2["ad_knn"], onvalue = 1, offvalue = 0)
ad_knn.grid(row=1, column=1, sticky=(tk.W + tk.E))

models2["ad_lof"] = tk.BooleanVar()
ad_lof = ttk.Checkbutton(ad_info, text="Local Outlier Factor (lof)", variable=models2["ad_lof"], onvalue = 1, offvalue = 0)
ad_lof.grid(row=1, column=2, sticky=(tk.W + tk.E))

models2["ad_svm"] = tk.BooleanVar()
ad_svm = ttk.Checkbutton(ad_info, text="One-class SVM detector (svm)", variable=models2["ad_svm"], onvalue = 1, offvalue = 0)
ad_svm.grid(row=1, column=3, sticky=(tk.W + tk.E))



models2["ad_pca"] = tk.BooleanVar()
ad_pca = ttk.Checkbutton(ad_info, text="Principal Component Analysis (pca)", variable=models2["ad_pca"], onvalue = 1, offvalue = 0)
ad_pca.grid(row=2, column=0, sticky=(tk.W + tk.E))

models2["ad_mcd"] = tk.BooleanVar()
ad_mcd = ttk.Checkbutton(ad_info, text="Minimum Covariance Determinant (mcd)", variable=models2["ad_mcd"], onvalue = 1, offvalue = 0)
ad_mcd.grid(row=2, column=1, sticky=(tk.W + tk.E))

models2["ad_sod"] = tk.BooleanVar()
ad_sod = ttk.Checkbutton(ad_info, text="Subspace Outlier Detection (sod)", variable=models2["ad_sod"], onvalue = 1, offvalue = 0)
ad_sod.grid(row=2, column=2, sticky=(tk.W + tk.E))

models2["ad_sos"] = tk.BooleanVar()
ad_sos = ttk.Checkbutton(ad_info, text="Stochastic Outlier Selection (sos)", variable=models2["ad_sos"], onvalue = 1, offvalue = 0)
ad_sos.grid(row=2, column=3, sticky=(tk.W + tk.E))



ttk.Button(ad_info, text="Select All", command=selectAllAD).grid(row=3, column=2, padx=5, pady=5, sticky=(tk.W + tk.E))
ttk.Button(ad_info, text="Unselect All", command=unselectAllAD).grid(row=3, column=3, padx=5, pady=5, sticky=(tk.W + tk.E))

# Show the window 
root.mainloop()