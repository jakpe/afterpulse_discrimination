import numpy as np                                     # Matlab like syntax for linear algebra and functions
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
import matplotlib as mpl
mpl.rcParams['font.size'] = 16                         # Set the general plotting font size                                 # Make the plots nicer to look at
from sklearn import metrics
import NN_tensorflow 
import fisher
import xgboost_dec_tree 
import SVM
import pandas as pd
from sklearn.model_selection import train_test_split


# Optimization - should remain this way, change next opt if optimization should be on
global opt
opt = False

# Data loading and splitting
Dataframe = pd.read_csv('parameters_w_label.csv')
training, test = train_test_split(Dataframe,test_size = 0.33, random_state=42)
training_label = training.label
test_label = test.label
training = training.drop(columns="label")
test = test.drop(columns="label")

# Figure preparation
fig, ax = plt.subplots(figsize=(12,10))
a = plt.axes([.36, .44, .57, .41])
a.set(xlim=(-0.01,0.23),
      ylim=(0.9,1.005),
      title="Zoomed in",)
a.grid(True)

ax.grid(True)
ax.tick_params(grid_linestyle='--')
a.tick_params(grid_linestyle='--')
ax.set(xlabel='False positive rate',
       ylabel='True positive rate',
       title='ROC Curve')

def roc_curve(prediction_0, prediction_1):
    """Gives a roc_curve based on prediction scores
    with a corresponding area """
    y_score = np.append(prediction_0, prediction_1)
    y_true1 = np.zeros(len(prediction_0))
    y_true2 = np.ones(len(prediction_1))
    y_true = np.append(y_true1, y_true2)
    y_true = [int(i) for i in y_true]

    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_score)
    roc_auc = metrics.auc(fpr, tpr)
    return fpr, tpr, roc_auc

def train_n_plot(training, training_label, test, test_label, Dataframe, ax, a):
    # NN
    model_NN, X_train, X_test, y_train, y_test, acc_NN = (
            NN_tensorflow.NN_clf(training,
                                    training_label,
                                    test, 
                                    test_label,
                                    Dataframe))
    y_pred_0n = model_NN.predict_proba(X_test[y_test == 0])[:, 1]
    y_pred_1n = model_NN.predict_proba(X_test[y_test == 1])[:, 1]
    fpr_NN, tpr_NN, area_NN = roc_curve(y_pred_0n, y_pred_1n)
    
    # Fisher
    clf, acc_lda = fisher.fisher_LDA(training, 
                            training_label,
                            test, 
                            test_label,
                            Dataframe)
    y_pred_0l = clf.predict_proba(test[test_label==0])[:,1]
    y_pred_1l = clf.predict_proba(test[test_label==1])[:,1]
    fpr_fis, tpr_fis, area_fis = roc_curve(y_pred_0l, y_pred_1l)
    
    # SVM
    model_svm, acc_svm = SVM.SVM_clf(training, training_label, 
                            test, test_label, Dataframe)
    y_pred_0s = model_svm.predict_proba(test[test_label == 0])[:, 1]
    y_pred_1s = model_svm.predict_proba(test[test_label == 1])[:, 1]
    fpr_svm, tpr_svm, area_svm = roc_curve(y_pred_0s, y_pred_1s)
    
    # XGboost
    model, acc_xgb = xgboost_dec_tree.xgb_clf(training, training_label, 
                                     test, test_label)
    y_pred_0x = model.predict_proba(test[test_label == 0])[:, 1]
    y_pred_1x = model.predict_proba(test[test_label == 1])[:, 1]
    fpr_XG, tpr_XG, area_XG = roc_curve(y_pred_0x, y_pred_1x)
    
# =============================================================================
#   Prediction plot (2x2) of all models 
# =============================================================================
    bins = 80
    figp, ((axp1, axp2), (axp3, axp4)) = plt.subplots(2, 2, 
          figsize=(12,12), sharex='col', sharey='row')
    mpl.rcParams['font.size'] = 16
    
    # LDA
    _ = axp1.hist(y_pred_0l, bins=bins, label='Signals',
                  edgecolor='darkgreen', facecolor='green', alpha=0.5)
    _ = axp1.hist(y_pred_1l, bins=bins, label='Afterpulses',
                  edgecolor='darkred', facecolor='red', alpha=0.5)
    axp1.grid(True, color='black', linestyle='--', linewidth=0.5, alpha=0.25)
    axp1.legend()
    axp1.set(#xlabel='prediction',
             ylabel='Frequency',
             yscale='log',
             title='LDA')
    
    # SVM
    _ = axp2.hist(y_pred_0s, bins=bins, label='Signals',
                  edgecolor='darkgreen', facecolor='green', alpha=0.5)
    _ = axp2.hist(y_pred_1s, bins=bins, label='Afterpulses',
                  edgecolor='darkred', facecolor='red', alpha=0.5)
    axp2.grid(True, color='black', linestyle='--', linewidth=0.5, alpha=0.25)
    axp2.legend()
    axp2.set(#xlabel='prediction',
             #ylabel='Frequency',
             yscale='log',
             title='SVM')
    
    # NN
    _ = axp3.hist(y_pred_0n, bins=bins, label='Signals',
                  edgecolor='darkgreen', facecolor='green', alpha=0.5)
    _ = axp3.hist(y_pred_1n, bins=bins, label='Afterpulses',
                  edgecolor='darkred', facecolor='red', alpha=0.5)
    axp3.grid(True, color='black', linestyle='--', linewidth=0.5, alpha=0.25)
    axp3.legend()
    axp3.set(xlabel='prediction',
             ylabel='Frequency',
             yscale='log',
             title='NN')
    
    # XGB
    _ = axp4.hist(y_pred_0x, bins=bins, label='Signals',
                  edgecolor='darkgreen', facecolor='green', alpha=0.5)
    _ = axp4.hist(y_pred_1x, bins=bins, label='Afterpulses',
                  edgecolor='darkred', facecolor='red', alpha=0.5)
    axp4.grid(True, color='black', linestyle='--', linewidth=0.5, alpha=0.25)
    axp4.legend()
    axp4.set(xlabel='prediction',
             #ylabel='Frequency',
             yscale='log',
             title='XGB')
    
    figp.savefig("prediction_models.pdf")
# =============================================================================
#   Adding to ROC curve  
# =============================================================================
    round_n = 4
    if opt:
        ax.plot(fpr_fis, tpr_fis, 
                label="Optimized Fisher's discriminant with area = " 
                + str(round(area_fis,round_n)),
                linestyle='-.', color='green')
        ax.plot(fpr_svm, tpr_svm, marker='x',
                label="Optimized SVM classifier with area = " 
                + str(round(area_svm,round_n)),
                linestyle='-', color='darkred')
        ax.plot(fpr_NN, tpr_NN, 
                label="Optimized neural network classifier with area = " 
                + str(round(area_NN,round_n)),
                linestyle='--', color='darkorange', marker='*')
        ax.plot(fpr_XG, tpr_XG, 
                label="Optimized XGBoost classifier with area = " 
                + str(round(area_XG,round_n)),
                linestyle=':', color='darkblue')
        # this is an inset axes over the main axes
        a.plot(fpr_fis, tpr_fis, 
                linestyle='-.', color='green')
        a.plot(fpr_svm, tpr_svm, 
                linestyle='-', color='darkred', marker='x')
        a.plot(fpr_NN, tpr_NN, 
                linestyle='-', color='darkorange', marker='*')
        a.plot(fpr_XG, tpr_XG, 
                linestyle=':', color='darkblue',)
        
    else:
        ax.plot(fpr_fis, tpr_fis, 
                label="Fisher's discriminant with area = " 
                + str(round(area_fis,round_n)),
                linestyle='-', color='green')
        ax.plot(fpr_svm, tpr_svm, 
                label="SVM classifier with area = " 
                + str(round(area_svm,round_n)),
                linestyle='-', color='darkred')
        ax.plot(fpr_NN, tpr_NN, 
                label="Neural network classifier with area = " 
                + str(round(area_NN,round_n)),
                linestyle='--', color='darkorange')
        ax.plot(fpr_XG, tpr_XG, 
                label="XGBoost classifier with area = " 
                + str(round(area_XG,round_n)),
                linestyle='-', color='darkblue')
    
        # this is an inset axes over the main axes
        a.plot(fpr_fis, tpr_fis, 
                linestyle='-', color='green')
        a.plot(fpr_svm, tpr_svm, 
                linestyle='-', color='darkred')
        a.plot(fpr_NN, tpr_NN, 
                linestyle='--', color='darkorange')
        a.plot(fpr_XG, tpr_XG, 
                linestyle='-', color='darkblue')

    print()
    print('Ratio of signals in test:', str(round(
        len(y_test[y_test == 0])/len(y_test)*100, 5)) + "%")
    print()
    
    
    
    
    # Make predictions for test data and printing accuracies
    print('[ACCURACY] LDA')
    print(str(round(acc_lda*100,3)) + "%")
    print()
    print('[ACCURACY] SVM')
    print(str(round(acc_svm*100,3)) + "%")
    print()
    print('[ACCURACY] NN')
    print(str(round(acc_NN*100,3)) + "%")
    print()
    print('[ACCURACY] XGboost')
    print(str(round(acc_xgb*100,3)) + "%")
    print() 
    
    print("Separation in standard deviation:")
    print("LDA: ", (np.mean(y_pred_1l) - np.mean(y_pred_0l)) /
          np.sqrt(np.std(y_pred_1l)**2 + np.std(y_pred_0l)**2))
    print("SVM: ", (np.mean(y_pred_1s) - np.mean(y_pred_0s)) /
          np.sqrt(np.std(y_pred_1s)**2 + np.std(y_pred_0s)**2))
    print("NN:  ", (np.mean(y_pred_1n) - np.mean(y_pred_0n)) /
          np.sqrt(np.std(y_pred_1n)**2 + np.std(y_pred_0n)**2))
    print("XGB: ", (np.mean(y_pred_1x) - np.mean(y_pred_0x)) /
          np.sqrt(np.std(y_pred_1x)**2 + np.std(y_pred_0x)**2))
    print()
    print("AUC:")
    print("LDA: ", area_fis)
    print("SVM: ", area_svm)
    print("NN:  ", area_NN)
    print("XGB: ", area_XG)
    print()
    print("Afterpulse discriminated at 99.9% correct classifying signals [%]:")
    print("LDA: ", min(tpr_fis[fpr_fis>0.001])*100)
    print("SVM: ", min(tpr_svm[fpr_svm>0.001])*100)
    print("NN:  ", min(tpr_NN[fpr_NN>0.001])*100)
    print("XGB: ", min(tpr_XG[fpr_XG>0.001])*100)
    print()
    
    if not opt:
        def get_weights(arb_model):
            _coef = [abs(i) for i in list(arb_model.coef_[0])]
            _wei = [i / sum(_coef) for i in _coef]
            return _wei
        clf_wei = get_weights(clf)
        svm_wei = get_weights(model_svm)
        
        figi, axi = plt.subplots(figsize=(16,10))
        width = 0.5
        df = pd.DataFrame(dict(graph=list(training.keys()),
                           XGB=list(model.feature_importances_), LDA=clf_wei,
                           SVM=svm_wei)) 
        df = df.iloc[::-1]
        corr_start, energy_end = 0-width*10, len(df)*2
        corr_form, form_energy = 7*width, len(df)*2-25*width
        axi.axhspan(corr_start, corr_form, facecolor='purple', alpha=0.4)
        axi.axhspan(corr_form, form_energy, facecolor='yellow', alpha=0.4)
        axi.axhspan(form_energy, energy_end, facecolor='cyan', alpha=0.4)
        axi.text(max(df.XGB) - 0.20, (corr_form-corr_start)/4,
                 "Correlation parameters",
                color='purple',
                alpha=0.6, fontsize=18)
        axi.text(max(df.XGB) - 0.20, (form_energy-corr_form)/1.5, 
                 "Shape of pulse parameters",
                color='orange',
                alpha=0.8, fontsize=18)
        axi.text(max(df.XGB) - 0.20, energy_end - (energy_end-form_energy)/1.5, 
                "Energy of pulse parameters", color='blue',
                alpha=0.6, fontsize=18)

        ind = np.arange(len(df))*2
        
        def format_func(value, tick_number):
            return str(round(value*100)) + "%"
        
        axi.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        
        axi.barh(ind + width, df.LDA, width, label='LDA',
                alpha=0.8, color='darkgreen', edgecolor='darkgreen');
        axi.barh(ind + 2*width, df.SVM, width, label='SVM',
                alpha=0.9, color='darkred', edgecolor='darkred');
        axi.barh(ind, df.XGB, width, label='XGB', 
                alpha=0.8, color='darkblue', edgecolor='darkblue');
                 
        axi.set(yticks=ind + width, yticklabels=df.graph, 
                ylim=[2*width - 2, len(df)*2],
                xlabel='Parameter importance')
        axi.legend(loc=(0.45, 0.32 ), framealpha=0.2, 
                          title="Models",
                          fontsize=14,
                          )
        
        for i in range(len(ind)):
            axi.text(list(df.XGB)[i]+0.001, ind[i]-0.21, 
                    str(round(list(df.XGB)[i]*100,3))+"%",
                     color='k', fontweight='bold', fontsize=9)
        for i in range(len(ind)):
            axi.text(list(df.LDA)[i]+0.001, ind[i]-0.21+width, 
                    str(round(list(df.LDA)[i]*100,3))+"%",
                     color='k', fontweight='bold', fontsize=9)
        for i in range(len(ind)):
            axi.text(list(df.SVM)[i]+0.001, ind[i]-0.21+width*2, 
                    str(round(list(df.SVM)[i]*100,3))+"%",
                     color='k', fontweight='bold', fontsize=9)

        
        figi.savefig("Feature_importance2.pdf")
    
    return ax, a

train_n_plot(training, training_label, test, test_label,
             Dataframe, ax, a)

# Boolean variable opt should be changed to true here if optimization should be on
opt = True

if opt:
    Dataframe_opt = pd.read_csv('parameters_w_label_opt.csv')
    training, test = train_test_split(Dataframe_opt,test_size = 0.33, random_state=42)
    training_label = training.label
    test_label = test.label
    training = training.drop(columns="label")
    test = test.drop(columns="label")
    train_n_plot(training, training_label, test, test_label, 
                 Dataframe_opt, ax, a)



ax.legend()
if opt:
    fig.savefig("roc_curve_opt.pdf")
else:
    fig.savefig("roc_curve.pdf")