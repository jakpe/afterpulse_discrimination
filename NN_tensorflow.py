from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

parameters = True
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
import matplotlib as mpl
mpl.rcParams['font.size'] = 15 
import numpy as np                                     # Matlab like syntax for linear algebra and functions
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split


def NN_clf(training, training_label, test, test_label, Dataframe):
    labels = np.array(Dataframe.label)
    Dataframe = Dataframe.drop(['label'], axis=1)
    X_train, X_test, y_train, y_test = training, test, training_label, test_label

    size = (np.shape(Dataframe)[1], 1)

    if not parameters:
        Dataframe_A = pd.read_csv("afterpulses.csv") 
        Dataframe_A.rename(columns={'a':'s',
                                         'isolated_time':'isolated_time'},
                                        inplace=True)
        Dataframe_S = pd.read_csv("signals.csv") 
        
        def reader(Df):
            waves = []
            for i in range(len(Df)):
            
                # Analysis
                wave = []
                list_wave = Df['s'][i]
                list_wave = list_wave.split(' ')
                for string in list_wave:
                    if string == '' or string == '[':
                        continue
                    string_temp = string.split('[')
                    if  len(string_temp) > 1:
                        string_temp = string_temp[1]
                    else:
                        string_temp = string_temp[0]
                    if string_temp == '':
                        continue
                    string_temp = string_temp.split(']')[0]
                    if string_temp == '':
                        continue
                    wave.append(float(string_temp))
                waves.append(wave)
            return np.array(waves)
        
        all_e = np.append(reader(Dataframe_S), reader(Dataframe_A))
        df = pd.DataFrame({'s':all_e})
        labels = np.append(np.zeros(len(Dataframe_S)),np.ones(len(Dataframe_A)))
        X_train, X_test, y_train, y_test = train_test_split(
                        df, labels, test_size=0.33, random_state=42)
        size = (np.shape(df)[1], 1)
    
    EPOCHS = 400
    X_train = np.expand_dims(X_train,2)
    X_test = np.expand_dims(X_test,2)
    
    def _trainmodel(X_train, y_train, X_test, y_test, EPOCHS):
        """ Returns model, saves figure of evolution
        and best iteration(s) step """
        from tensorflow import set_random_seed
        set_random_seed(42)
        
        # Setting variables
        optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99,
                                          epsilon=None, decay=0.0, amsgrad=False)
        
        first_layer_neurons = 50
        second_layer_neurons = 60 
        third_layer_neurons = second_layer_neurons
        
        
        model_NN = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=size),
            tf.keras.layers.Dense(first_layer_neurons, activation=tf.tanh),
            tf.keras.layers.Dense(second_layer_neurons, activation=tf.tanh),
            tf.keras.layers.Dense(third_layer_neurons, activation=tf.tanh),
            tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)
        ])
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', 
                                                   patience=25)
        
        model_NN.compile(optimizer=optimizer, 
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'],)
        
        H1 = model_NN.fit(X_train, y_train,
        	validation_data=(X_test, y_test),
        	epochs=EPOCHS, verbose=1,
          callbacks=[callback])
        
        H2 = model_NN.fit(X_train, y_train,
        	validation_data=(X_test, y_test),
        	epochs=EPOCHS, verbose=1)
        
        test_loss_NN, test_acc_NN = model_NN.evaluate(X_test, 
                                                         y_test)
    
        # Saving evolution of metrics throughout the iterations
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(np.arange(0, EPOCHS), H2.history["loss"], 
                label="Train loss", color='darkblue')
        ax.plot(np.arange(0, EPOCHS), H2.history["val_acc"],
                label="Validation accuracy", color='darkred')
        ax.plot(np.arange(0, EPOCHS), H2.history["acc"], 
                label="Train accuracy", color='darkgreen')
        ax.plot(np.arange(0, EPOCHS), H2.history["val_loss"], 
                label="Validation loss", color='darkorange')
        ax.plot([len(H1.history["val_acc"])-1, len(H1.history["val_acc"])-1],
                 [0,1], '--r')
        ax.set(xlabel="Iteration",
               ylabel=("Loss/Accuracy"),
               ylim=(-0.01, 1.01)
               )
        ax.grid(True)
        ax.legend(loc=(0.3,0.4))
        fig.savefig('evolutionNN' + str(EPOCHS) + '.pdf')
        
        return model_NN, test_acc_NN
    
    model_NN, test_acc_NN = _trainmodel(
            X_train, y_train, X_test, y_test, EPOCHS)

 
    return model_NN, X_train, X_test, y_train, y_test, test_acc_NN
