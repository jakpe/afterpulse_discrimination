import numpy as np                                     # Matlab like syntax for linear algebra and functions
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
import pandas as pd
from numpy.linalg import inv
from sklearn.model_selection import train_test_split

#Dataframe2 = pd.read_csv('parameters_w_label.csv')


def fisher_LDA(training, training_label, test, test_label, Dataframe):
    
    
#    # With label
#    D_sig_true = Dataframe[Dataframe.label == 0]
#    D_af_true = Dataframe[Dataframe.label == 1]
#    
#    # No label 
#    D_sig = D_sig_true.drop(columns="label")
#    D_af = D_af_true.drop(columns="label")
#    
#    # Normalizing 
#    for key in D_sig:
#        D_sig[key] = D_sig[key] / np.mean(D_sig[key])
#        D_af[key] = D_af[key] / np.mean(D_af[key])
#    
#    # Switching to language of Fisher
#    spec_A = D_sig
#    spec_B = D_af
#    
#    if type(spec_A) == pd.core.frame.DataFrame:
#          speca, specb = [], []
#          for key in spec_A:
#              if key != 'face':
#                  speca.append(np.array(spec_A[key]))
#          for key in spec_B:
#              if key != 'face':
#                  specb.append(np.array(spec_B[key]))
#          spec_A, spec_B = speca, specb
#      
#    # Construct mean vectors
#    mu_A, mu_B = [], []
#      
#    for i in range(len(spec_A)):
#          mu_A.append(np.mean(spec_A[i]))
#          mu_B.append(np.mean(spec_B[i]))
#    mu_A = np.array(mu_A) 
#    mu_B = np.array(mu_B) 
#    mu_AB = mu_A - mu_B
#    
#    # Covariance and weight
#    cov_A = np.cov(spec_A) 
#    cov_B = np.cov(spec_B) 
#    cov_sum_inv = inv(cov_A + cov_B)
#    wei = np.dot(cov_sum_inv,mu_AB)
#    
#    # Normalization
#    w0 = - np.dot(wei,mu_A)
#    C = 1/(np.dot(wei,mu_B) + w0)
#    
#    # Dot product of data and weights
#    dot_A = np.ones(len(spec_A[0]))
#    dot_B = np.ones(len(spec_B[0]))
#    
#    for i in range(len(spec_A[0])):
#        vec_A = np.ones(len(spec_A))
#        for l in range(len(spec_A)):
#            vec_A[l] = spec_A[l][i]
#        dot_A[i] = np.dot(wei,vec_A)
#        
#    for i in range(len(spec_B[0])):
#        vec_B = np.ones(len(spec_B))
#        for l in range(len(spec_B)):
#            vec_B[l] = spec_B[l][i] 
#        dot_B[i] = np.dot(wei,vec_B)    
#        
#    
#    # Finally Fisher
#    fisher_A = C * (w0 + dot_A)
#    fisher_B = C * (w0 + dot_B)
#    
#    # Quality of seperation
#    delta = (np.mean(fisher_A)-np.mean(fisher_B))**2/(np.std(fisher_A)**2
#             + np.std(fisher_B)**2)
#    
#    # Threshold
#    Fmin, Fmax = min(fisher_A), max(fisher_B)
#    histFA, bin_edges = np.histogram(fisher_A, bins=250, range=(Fmin,Fmax))
#    histFB, bin_edges = np.histogram(fisher_B, bins=250, range=(Fmin,Fmax))
#    bins_distance = bin_edges[1] - bin_edges[0]
#    for i in range(len(histFB)):
#        if histFB[i] > histFA[i]:
#            threshold = Fmin + i*bins_distance
#            break
#    
#    threshold = 0.5
#    
#    # Alpha (Type 1) and Beta (Type 2)
#    alpha = len([i for i in fisher_A if i > threshold])/len(fisher_A)
#    beta = len([i for i in fisher_B if i < threshold])/len(fisher_B)
#    print('goodness of sep')
#    print(delta)
#    bins = 200
#    
#    fig, ax = plt.subplots(figsize=(6,5))
#    histA, binsA, _ = ax.hist(fisher_A, label = 'Signal', bins=bins,
#                              histtype = 'step', 
#                              range=(min(fisher_A),max(fisher_B)), 
#                              color= 'darkgreen')
#    histB, binsB, _ = ax.hist(fisher_B, label ='Afterpulse', bins=bins, 
#                              histtype='step',
#                              range=(min(fisher_A),max(fisher_B)),
#                              color = 'darkred')
#    ax.set(xlabel="Fisher's discriminant",
#           ylabel='Freqency per ' + str(round((binsA[1]-binsA[0]),3)))
#    ax.legend(loc='upper left', frameon=True, framealpha=0.2, fancybox=True)
    
    # Imported Model
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    for key in training:
        training[key] = training[key] / np.mean(training[key])
        test[key] = test[key] / np.mean(test[key])
    
    clf.fit(training, training_label)
    acc = clf.score(test, test_label)
    
    return clf, acc