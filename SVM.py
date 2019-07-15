import numpy as np                                     # Matlab like syntax for linear algebra and functions
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import time
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
import matplotlib as mpl
mpl.rcParams['font.size'] = 15 
from sklearn.metrics import accuracy_score

def SVM_clf(training, training_label, test, test_label, df):
    development = False
    
    if development:
        max_iter = 10**(7)
    else:
        max_iter = 10**(10)
        
    # Model
    start = time.time()
    print()
    print("SVM training begins")
    fixed_params = {'probability':True, 
            'max_iter':max_iter,
            'random_state':42,
            'kernel':'linear'
            }
    model_svm = svm.SVC(fixed_params)
    
    # Grid search and tuning with inspiration from
    # https://anaconda.org/hhllcks/monsters/notebook
    grid_param = {
            'cache_size': [200,1000],
            'gamma': np.linspace(0.02, 2.0, 7),
            'C': np.linspace(0.5, 1.5, 5),
                  }
    # For developing grid_param is redefined 
    # This should be outcommented when doing the full analysis
    if development:
        grid_param = {
                'cache_size': [200,1000],
                'gamma':np.linspace(0.02, 2.0, 1),
                'C': np.linspace(0.5, 1.5, 1),
                      }
    
    grid_search = GridSearchCV(estimator=model_svm,  
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=-1)
    grid_search.fit(training, training_label)  
    best_parameters = grid_search.best_params_  
    
    print()
    print("Best Hyperparameters for model:")
    print(best_parameters)  
    print()
    expmin, expmax = 1, 5.5
    if development:
        iterations = [10**(4)]
    else:
        iterations = [10**(i) for i in np.linspace(expmin, expmax, 200)]
    
    train_acc, test_acc = [], []
    for ite in iterations:
        model_svm = svm.SVC(
                    gamma=best_parameters['gamma'],
                    C=best_parameters['C'],
                    probability=True, 
                    max_iter=ite,               
                    random_state=42,
                    kernel='linear')
        model_svm.fit(training, training_label)
        y_predtr = model_svm.predict(training)
        predictionstr = [round(value) for value in y_predtr]
        y_predte = model_svm.predict(test)
        predictionste = [round(value) for value in y_predte]
        acctr = accuracy_score(training_label, predictionstr)
        accte = accuracy_score(test_label, predictionste)
        train_acc.append(acctr)
        test_acc.append(accte)
 
    # Saving evolution of metrics throughout the iterations
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(iterations, train_acc,
            label="Train accuracy", color='darkgreen')
    ax.plot(iterations, test_acc,
            label="Test accuracy", color='darkred')

    ax.plot([iterations[np.argmax(test_acc)], 
             iterations[np.argmax(test_acc)]],
             [min(test_acc),1], '--r')
    ax.set(xlabel="Iteration",
           ylabel=("Accuracy score"),
           #ylim=(-0.01, 1.01),
           xscale='log',
           )
    ax.grid(True)
    ax.legend()#loc=(0.3,0.4))
    fig.savefig('evolutionSVM' + str(int(expmax)) + '.pdf')

    model_svm = svm.SVC(
                    gamma=best_parameters['gamma'],
                    C=best_parameters['C'],
                    probability=True, 
                    max_iter=iterations[np.argmax(test_acc)],               
                    random_state=42,
                    kernel='linear')
    model_svm.fit(training, training_label)
    y_pred = model_svm.predict(test)
    predictions = [round(value) for value in y_pred]
    acc = accuracy_score(test_label, predictions)
    print("SVM training took " + str(time.time()-start) + "s")

   
    
    return model_svm, acc