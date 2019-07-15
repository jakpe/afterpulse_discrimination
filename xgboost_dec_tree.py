from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
import matplotlib as mpl
import numpy as np
mpl.rcParams['font.size'] = 15 

def xgb_clf(training, training_label, test, test_label):
    
    # Hyperparameters sat with inspiration from:
    # https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    rounds = 400
    eta = 0.2 # 0.01-2
    max_depth = 7 # 3-10 default: 6
    gamma = 0.1 # default 0, but should be tuned
    
    # Evaluation
    eval_set = [(training, training_label), (test, test_label)]
    
    # Construct model and train
    model = XGBClassifier(seed=42, eta=eta,
                          max_depth=max_depth, gamma=gamma, 
                          n_estimators=rounds, verbose=False)
    model.fit(training, training_label, eval_metric='logloss',
              eval_set=eval_set)  
    results = model.evals_result()
    model.fit(training, training_label, eval_metric='logloss',
              eval_set=eval_set, early_stopping_rounds=5, verbose=False)  
    EPOCHS = len(results["validation_0"]['logloss'])

    # Saving evolution of metrics throughout the iterations
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(np.arange(0, EPOCHS), results["validation_0"]['logloss'],
            label="Train log loss", color='darkblue')
    ax.plot(np.arange(0, EPOCHS), results["validation_1"]['logloss'],
            label="Test log loss", color='darkorange')

    ax.plot([model.best_iteration, model.best_iteration],
             [0,1], '--r')
    ax.set(xlabel="Iteration",
           ylabel=("Logarithmic loss"),
           #ylim=(-0.01, 1.01)
           )
    ax.grid(True)
    ax.legend()#loc=(0.3,0.4))
    fig.savefig('evolutionXGB' + str(EPOCHS) + '.pdf')
    
    y_pred = model.predict(test)
    predictions = [round(value) for value in y_pred]
    acc = accuracy_score(test_label,predictions)

    return model, acc