# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 22:45:19 2023

@author: UMONS - 532807
"""

import tkinter as tk
from tkinter import ttk

variables = dict()
models1 = dict()
models2 = dict()

root = tk.Tk()
root.title('Ai Toolbox - Machine Learning - Supervised Learning')
root.columnconfigure(0, weight=1)

mc = ttk.Frame(root)
mc.grid(padx=10, pady=10, sticky=(tk.W + tk.E))
mc.columnconfigure(0, weight=1)


def selectAllRegression():
    for i in models1.keys():
        models1[i].set(1)

    
def unselectAllRegression():
    for i in models1.keys():
        models1[i].set(0) 
    

regression_info = ttk.LabelFrame(mc, text='Regression')
regression_info.grid(padx=5, pady=5, sticky=(tk.W + tk.E))
for i in range(4):
    regression_info.columnconfigure(i, weight=1)
    
    
models1["regression_lr"] = tk.BooleanVar()
regression_lr = ttk.Checkbutton(regression_info, text="Linear Regression (lr)", variable=models1["regression_lr"], onvalue = 1, offvalue = 0)
regression_lr.grid(row=0, column=0, sticky=(tk.W + tk.E))

models1["regression_lasso"] = tk.BooleanVar()
regression_lasso = ttk.Checkbutton(regression_info, text="Lasso Regression (lasso)", variable=models1["regression_lasso"], onvalue = 1, offvalue = 0)
regression_lasso.grid(row=0, column=1, sticky=(tk.W + tk.E))

models1["regression_ridge"] = tk.BooleanVar()
regression_ridge = ttk.Checkbutton(regression_info, text="Ridge Regression (ridge)", variable=models1["regression_ridge"], onvalue = 1, offvalue = 0)
regression_ridge.grid(row=0, column=2, sticky=(tk.W + tk.E))

models1["regression_en"] = tk.BooleanVar()
regression_en = ttk.Checkbutton(regression_info, text="Elastic Net (en)", variable=models1["regression_en"], onvalue = 1, offvalue = 0)
regression_en.grid(row=0, column=3, sticky=(tk.W + tk.E))



models1["regression_lar"] = tk.BooleanVar()
regression_lar = ttk.Checkbutton(regression_info, text="Least Angle Regression (lar)", variable=models1["regression_lar"], onvalue = 1, offvalue = 0)
regression_lar.grid(row=1, column=0, sticky=(tk.W + tk.E))

models1["regression_llar"] = tk.BooleanVar()
regression_llar = ttk.Checkbutton(regression_info, text="Lasso Least Angle Regression (llar)", variable=models1["regression_llar"], onvalue = 1, offvalue = 0)
regression_llar.grid(row=1, column=1, sticky=(tk.W + tk.E))

models1["regression_omp"] = tk.BooleanVar()
regression_omp = ttk.Checkbutton(regression_info, text="Orthogonal Matching Pursuit (omp)", variable=models1["regression_omp"], onvalue = 1, offvalue = 0)
regression_omp.grid(row=1, column=2, sticky=(tk.W + tk.E))

models1["regression_br"] = tk.BooleanVar()
regression_br = ttk.Checkbutton(regression_info, text="Bayesian Ridge (br)", variable=models1["regression_br"], onvalue = 1, offvalue = 0)
regression_br.grid(row=1, column=3, sticky=(tk.W + tk.E))



models1["regression_ard"] = tk.BooleanVar()
regression_ard = ttk.Checkbutton(regression_info, text="Automatic Relevance Determination (ard)", variable=models1["regression_ard"], onvalue = 1, offvalue = 0)
regression_ard.grid(row=2, column=0, sticky=(tk.W + tk.E))

models1["regression_par"] = tk.BooleanVar()
regression_par =  ttk.Checkbutton(regression_info, text="Passive Aggressive Regressor (par)", variable=models1["regression_par"], onvalue = 1, offvalue = 0)
regression_par.grid(row=2, column = 1, sticky=(tk.W + tk.E))

models1["regression_ransac"] = tk.BooleanVar()
regression_ransac = ttk.Checkbutton(regression_info, text="Random Sample Consensus (ransac)", variable=models1["regression_ransac"], onvalue = 1, offvalue = 0)
regression_ransac.grid(row=2, column = 2, sticky=(tk.W + tk.E))

models1["regression_tr"] = tk.BooleanVar()
regression_tr = ttk.Checkbutton(regression_info, text="TheilSen Regressor (tr)", variable=models1["regression_tr"], onvalue = 1, offvalue = 0)
regression_tr.grid(row=2, column=3, sticky=(tk.W + tk.E))



models1["regression_huber"] = tk.BooleanVar()
regression_huber = ttk.Checkbutton(regression_info, text="Huber Regressor (huber)", variable=models1["regression_huber"], onvalue = 1, offvalue = 0)
regression_huber.grid(row=3, column=0, sticky=(tk.W + tk.E))

models1["regression_kr"] = tk.BooleanVar()
regression_kr = ttk.Checkbutton(regression_info, text="Kernel Ridge (kr)", variable=models1["regression_kr"], onvalue = 1, offvalue = 0)
regression_kr.grid(row=3, column=1, sticky=(tk.W + tk.E))

models1["regression_svm"] = tk.BooleanVar()
regression_svm = ttk.Checkbutton(regression_info, text="Support Vector Regression (svm)", variable=models1["regression_svm"], onvalue = 1, offvalue = 0)
regression_svm.grid(row=3, column=2, sticky=(tk.W + tk.E))

models1["regression_knn"] = tk.BooleanVar()
regression_knn = ttk.Checkbutton(regression_info, text="K Neighbors Regressor (knn)", variable=models1["regression_knn"], onvalue = 1, offvalue = 0)
regression_knn.grid(row=3, column=3, sticky=(tk.W + tk.E))



models1["regression_dt"] = tk.BooleanVar()
regression_dt = ttk.Checkbutton(regression_info, text="Decision Tree Regressor (dt)", variable=models1["regression_dt"], onvalue = 1, offvalue = 0)
regression_dt.grid(row=4, column=0, sticky=(tk.W + tk.E))

models1["regression_rf"] = tk.BooleanVar()
regression_rf = ttk.Checkbutton(regression_info, text="Random Forest Regressor (rf)", variable=models1["regression_rf"], onvalue = 1, offvalue = 0)
regression_rf.grid(row=4, column=1, sticky=(tk.W + tk.E))

models1["regression_et"] = tk.BooleanVar()
regression_et = ttk.Checkbutton(regression_info, text="Extra Trees Regressor (et)", variable=models1["regression_et"], onvalue = 1, offvalue = 0)
regression_et.grid(row=4, column=2, sticky=(tk.W + tk.E))

models1["regression_ada"] = tk.BooleanVar()
regression_ada = ttk.Checkbutton(regression_info, text="AdaBoost Regressor (ada)", variable=models1["regression_ada"], onvalue = 1, offvalue = 0)
regression_ada.grid(row=4, column=3, sticky=(tk.W + tk.E))



models1["regression_gbr"] = tk.BooleanVar()
regression_gbr = ttk.Checkbutton(regression_info, text="Gradient Boosting Regressor (gbr)", variable=models1["regression_gbr"], onvalue = 1, offvalue = 0)
regression_gbr.grid(row=5, column=0, sticky=(tk.W + tk.E))

models1["regression_mlp"] = tk.BooleanVar()
regression_mlp = ttk.Checkbutton(regression_info, text="MLP Regressor (mlp)", variable=models1["regression_mlp"], onvalue = 1, offvalue = 0)
regression_mlp.grid(row=5, column=1, sticky=(tk.W + tk.E))

models1["regression_xgboost"] = tk.BooleanVar()
regression_xgboost = ttk.Checkbutton(regression_info, text="Extreme Gradient Boosting (xgboost)", variable=models1["regression_xgboost"], onvalue = 1, offvalue = 0)
regression_xgboost.grid(row=5, column=2, sticky=(tk.W + tk.E))

models1["regression_lightgbm"] = tk.BooleanVar()
regression_lightgbm = ttk.Checkbutton(regression_info, text="Light Gradient Boosting Machine (lightgbm)", variable=models1["regression_lightgbm"], onvalue = 1, offvalue = 0)
regression_lightgbm.grid(row=5, column=3)



models1["regression_catboost"] = tk.BooleanVar()
regression_catboost = ttk.Checkbutton(regression_info, text="CatBoost (catboost)", variable=models1["regression_catboost"], onvalue = 1, offvalue = 0)
regression_catboost.grid(row=6, column=0, sticky=(tk.W + tk.E))

models1["regression_dummy"] = tk.BooleanVar()
regression_dummy = ttk.Checkbutton(regression_info, text="Dummy Regressor (dummy)", variable=models1["regression_dummy"], onvalue=1, offvalue=0)
regression_dummy.grid(row=6, column=1, sticky=(tk.W + tk.E))

models1["regression_bagging"] = tk.BooleanVar()
regression_bagging = ttk.Checkbutton(regression_info, text="Bagging Regressor (bagging)", variable=models1["regression_bagging"], onvalue=1, offvalue=0)
regression_bagging.grid(row=6, column=2, sticky=(tk.W + tk.E))

models1["regression_stacking"] = tk.BooleanVar()
regression_stacking = ttk.Checkbutton(regression_info, text="Stacking Regressor (stacking)", variable=models1["regression_stacking"], onvalue=1, offvalue=0)
regression_stacking.grid(row=6, column=3, sticky=(tk.W + tk.E))



models1["regression_voting"] = tk.BooleanVar()
regression_voting = ttk.Checkbutton(regression_info, text="Voting Regressor (voting)", variable=models1["regression_voting"], onvalue=1, offvalue=0)
regression_voting.grid(row=7, column=0, sticky=(tk.W + tk.E))



ttk.Button(regression_info, text="Select All", command=selectAllRegression).grid(row=8, column=2, padx=5, pady=5, sticky=(tk.W + tk.E))
ttk.Button(regression_info, text="Unselect All", command=unselectAllRegression).grid(row=8, column=3, padx=5, pady=5, sticky=(tk.W + tk.E))


def selectAllClassification():
    for i in models2.keys():
        models2[i].set(1)
        
        
def unselectAllClassification():
    for i in models2.keys():
        models2[i].set(0)
                
        
classification_info = ttk.LabelFrame(mc, text='Classification')
classification_info.grid(padx=5, pady=5, sticky=(tk.W + tk.E))
for i in range(4):
    classification_info.columnconfigure(i, weight=1)

 
models2["classification_lr"] = tk.BooleanVar()
classification_lr = ttk.Checkbutton(classification_info, text="Logistic Regression (lr)", variable = models2["classification_lr"], onvalue = 1, offvalue = 0)
classification_lr.grid(row=0, column=0, sticky=(tk.W + tk.E))

models2["classification_knn"] = tk.BooleanVar()
classification_knn = ttk.Checkbutton(classification_info, text="K Neightbors Classifier (knn)", variable=models2["classification_knn"], onvalue = 1, offvalue = 0)
classification_knn.grid(row=0, column=1, sticky=(tk.W + tk.E))

models2["classification_nb"] = tk.BooleanVar()
classification_nb = ttk.Checkbutton(classification_info, text="Gaussian Naive Bayes (nb)", variable=models2["classification_nb"], onvalue = 1, offvalue = 0)
classification_nb.grid(row=0, column=2, sticky=(tk.W + tk.E))

models2["classification_dt"] = tk.BooleanVar()
classification_dt = ttk.Checkbutton(classification_info, text="Decision Tree Classifier (dt)", variable=models2["classification_dt"], onvalue = 1, offvalue = 0)
classification_dt.grid(row=0, column=3, sticky=(tk.W + tk.E))



models2["classification_svm"] = tk.BooleanVar()
classification_svm = ttk.Checkbutton(classification_info, text="SVM - Linear Kernel (svm)", variable = models2["classification_svm"], onvalue = 1, offvalue = 0)
classification_svm.grid(row=1, column = 0, sticky=(tk.W + tk.E))

models2["classification_rbfsvm"] = tk.BooleanVar()
classification_rbfsvm = ttk.Checkbutton(classification_info, text="SVM - Radial Kernel (rbfsvm)", variable=models2["classification_rbfsvm"], onvalue = 1, offvalue = 0)
classification_rbfsvm.grid(row=1, column=1, sticky=(tk.W + tk.E))

models2["classification_gpc"] = tk.BooleanVar()
classification_gpc = ttk.Checkbutton(classification_info, text="Gaussian Process Classifier (gpc)", variable=models2["classification_gpc"], onvalue = 1, offvalue = 0)
classification_gpc.grid(row=1, column=2, sticky=(tk.W + tk.E))

models2["classification_mlp"] = tk.BooleanVar()
classification_mlp = ttk.Checkbutton(classification_info, text="MLP Classifier (mlp)", variable=models2["classification_mlp"], onvalue = 1, offvalue = 0)
classification_mlp.grid(row=1, column=3, sticky=(tk.W + tk.E))



models2["classification_ridge"] = tk.BooleanVar()
classification_ridge = ttk.Checkbutton(classification_info, text="Ridge Classifier (ridge)", variable=models2["classification_ridge"], onvalue = 1, offvalue = 0)
classification_ridge.grid(row=2, column=0, sticky=(tk.W + tk.E))

models2["classification_rf"] = tk.BooleanVar()
classification_rf = ttk.Checkbutton(classification_info, text="Random Forest Classifier (rf)", variable=models2["classification_rf"], onvalue = 1, offvalue = 0)
classification_rf.grid(row=2, column=1, sticky=(tk.W + tk.E))

models2["classification_qda"] = tk.BooleanVar()
classification_qda = ttk.Checkbutton(classification_info, text="Quadratic Discriminant Analysis (qda)", variable=models2["classification_qda"], onvalue = 1, offvalue = 0)
classification_qda.grid(row=2, column=2, sticky=(tk.W + tk.E))

models2["classification_ada"] = tk.BooleanVar()
classification_ada = ttk.Checkbutton(classification_info, text="Ada Boost Classifier (ada)", variable=models2["classification_ada"], onvalue = 1, offvalue = 0)
classification_ada.grid(row=2, column=3, sticky=(tk.W + tk.E))



models2["classification_gbc"] = tk.BooleanVar()
classification_gbc = ttk.Checkbutton(classification_info, text="Grandient Boost Classifier (bgc)", variable=models2["classification_gbc"], onvalue = 1, offvalue = 0)
classification_gbc.grid(row = 3, column=0, sticky=(tk.W + tk.E))

models2["classification_lda"] = tk.BooleanVar()
classification_lda = ttk.Checkbutton(classification_info, text="Linear Discriminant Analysis (lda)", variable=models2["classification_lda"], onvalue = 1, offvalue = 0)
classification_lda.grid(row=3, column=1, sticky=(tk.W + tk.E))

models2["classification_et"] = tk.BooleanVar()
classification_et = ttk.Checkbutton(classification_info, text="Extra Trees Classifier (et)", variable=models2["classification_et"], onvalue = 1, offvalue = 0)
classification_et.grid(row=3, column=2, sticky=(tk.W + tk.E))

models2["classification_xgboost"] = tk.BooleanVar()
classification_xgboost = ttk.Checkbutton(classification_info, text="Extreme Gradient Boosting (xgboost)", variable=models2["classification_xgboost"], onvalue=1, offvalue=0)
classification_xgboost.grid(row=3, column=3, sticky=(tk.W + tk.E))



models2["classification_lightgbm"] = tk.BooleanVar()
classification_lightgbm = ttk.Checkbutton(classification_info, text="light Gradient Boosting Machine (lightgbm)", variable=models2["classification_lightgbm"], onvalue=1, offvalue=0)
classification_lightgbm.grid(row=4, column=0, sticky=(tk.W + tk.E))

models2["classification_catboost"] = tk.BooleanVar()
classification_catboost = ttk.Checkbutton(classification_info, text="CatBoost Classifier (catboost)", variable=models2["classification_catboost"], onvalue=1, offvalue=0)
classification_catboost.grid(row=4, column=1, sticky=(tk.W + tk.E))

models2["classification_dummy"] = tk.BooleanVar()
classification_dummy = ttk.Checkbutton(classification_info, text="Dummy Classifier (dummy)", variable=models2["classification_dummy"], onvalue=1, offvalue=0)
classification_dummy.grid(row=4, column=2, sticky=(tk.W + tk.E))



ttk.Button(classification_info, text="Select All", command=selectAllClassification).grid(row=5, column=2, padx=5, pady=5, sticky=(tk.W + tk.E))
ttk.Button(classification_info, text="Unselect All", command=unselectAllClassification).grid(row=5, column=3, padx=5, pady=5, sticky=(tk.W + tk.E))


# Show the window 
root.mainloop()