import preprocessing as pp 
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from timeseries_to_supervised import series_to_supervised
import matplotlib.pyplot as plt 
import pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.metrics import r2_score
import pca
import visualizations as vs 
import clustering
import metrics
#import tensorflow as tf 
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization
from keras.layers import Dropout, concatenate
from keras.optimizers import Adam
import matplotlib.pyplot as plt 

def reframe(df, n_timestamps, n_pca):
    """
    Reframes dataframe with shape (1;n_timestamps * n_pca) into (n_timestamps, n_pca)
    Arguments:
        df - input dataframe to be reframed
        n_timestamps - number of timestamps in a dataframe (most probably you are using this
            function after sliding_windows() so just put here amount of timestamps predicted)
        n_pca - number of principal components or variables inside the dataframe
    Returns:
        reframed_df - reframed dataframe 
    """
    reframed_df = pd.DataFrame()
    df_arr = np.squeeze(df.values)
    for pc in range(1, n_pca + 1):
        dfN = df_arr[[(pc-1 + i*4) for i in range(0, n_timestamps)]]

        reframed_df = pd.concat([reframed_df, pd.DataFrame(data=dfN, columns=['Principal Component ' + str(pc)])], axis=1)

    return reframed_df

def sliding_windows(data_df, n_predictions, model):
    """
    Predict next n_predictions values using sliding windows method.
    Cannot predict more timestamps than there is in input dataframe (not a bug but a feature, yeap)
    Arguments:
        data_df - dataframe with a data to base predictions on 
        n_predictions - how many values to predict on
        model - sklearn model used for predictions
    Returns:
        pred_df - dataframe with predicted values
    """
    pred_df = data_df.reset_index(drop=True)
    for i in range(n_predictions):
        p = model.predict(pred_df)
        df = pd.DataFrame(data=p)
        pred_df = pd.concat([pred_df, df], axis=1, ignore_index=True)
        pred_df = pred_df.drop(columns=[pred_df.columns.values[0],
                                        pred_df.columns.values[1],
                                        pred_df.columns.values[2],
                                        pred_df.columns.values[3]], axis=1)

    
    return pred_df

def get_test_data(index, test_df, n_pca, length_pred):
    """
    Returns two dataframes with input and expected ouput values for testing.
    Arguments:
        index - starting point for selecting data for testing
        test_df - dataframe with testing data in a supervised form
        n_pca - number of principal components or usual variables in the dataset
        length_pred - length of the predictions in amount of timestamps. If it is left 0, then there won't be any y_labels for last test_x samples 
    Returns: 
        test_x - dataframe with input data for testing
        test_y - dataframe with expected output values for each test sample from test_x
    """
    x = test_df[test_df.columns[:-n_pca]] #select all variables except for last n_pca 
    y = test_df[test_df.columns[-n_pca:]]

    test_x = x[index + 1:-length_pred]

    y = y[index :]
    data = list()
    for i in range(0, test_x.shape[0]):
        data.append(y[i:length_pred + i])
    
    test_y = pd.DataFrame(data=data)
    
    return test_x, test_y

def test_frame(test_x, test_y, model, timestamps_to_skip, n_pred):
    """
    Applies sliding window method on the whole dataframe to make predictions along whole dataframe
    Arguments:
        test_x - dataframe with input test data
        test_y - dataframe with labels for testing
        model - pretrained model used for predictions
        timestamps_to_skip - determines how many timestamps are skipped before next predictions.
            if it is 0, then the model effectevly predicts only 1 next hour, so just use the same value as the length of the prediction
        n_pred - number of consecutive predictions in sliding windows, works best when the same as timestamps_to_skip 
    Returns:
        pred_df - dataframe with predicted values, doesn't require reframing
        test_labels - labels for testing
    """ 

    #calculating number of prediction as multiple of n_pred 
    N = test_x.shape[0] // n_pred #floor division
    pred_df = pd.DataFrame()
    y_indx = list()
    for i in range(0, N):
        pred_n = sliding_windows(test_x[i*timestamps_to_skip:i*timestamps_to_skip+1], 96, model)
        r_n = reframe(pred_n, n_pred, 4)
        pred_df = pred_df.append(r_n, ignore_index=True)
        y_indx.append(i * timestamps_to_skip)
    test_labels = build_ylabels(y_indx, test_y)

    return pred_df, test_labels

def build_ylabels(y_indx, data):
    """
    Builds appropriate dataframe with labels based on indices
    Arguments:
        y_indx - indices to be included
        data - dataframe with all labels
    Returns:
        labels - dataframe with labels
    """
    labels = pd.DataFrame()

    for i in y_indx:
        labels = labels.append(data[0][i])
    
    return labels
        
def plot_predictions(data_df, labels_df, cop_df):
    """
    Plots predictions, their expected values, COP of the heat pump and predicted classes
    Arguments:
        data_df - dataframe with predictions and clustering results
        labels_df - dataframe with labels 
        cop_df - dataframe with COP of the heat pump
    """

    pred_length = data_df.shape[0]
    cop_length = cop_df.shape[0]

    plt.title('Principal component analysis: predictions/true labels')
    for plot in range(1, 5):
        plt.subplot(6, 2, plot*2 - 1)
        plt.plot([i for i in range(pred_length)], data_df['Principal Component ' + str(plot)])
        plt.ylabel('Principal Component ' + str(plot) + 'Predicted')
        plt.subplot(6, 2, plot*2)
        plt.plot([i for i in range(pred_length)], labels_df['Principal Component ' + str(plot)])
        plt.ylabel('Principal Component ' + str(plot) + 'True')

    plt.subplot(6, 2, 9)
    plt.plot([i for i in range(pred_length)], data_df['class'])
    plt.ylabel('Class predicted')

    plt.subplot(6, 2, 10)
    plt.plot([i for i in range(pred_length)], labels_df['class'])
    plt.ylabel('Class true label')

    plt.subplot(6, 1, 6)
    plt.plot([i for i in range(cop_length)], cop_df)
    plt.ylabel('COP')

    plt.show()

def open_model(name):
    """
    Opens saved model
    Arguments:
    name - model's name 
    Returns:
    model - sklearn model
    """
    with open(name, 'rb') as pickle_model:
        model = pickle.load(pickle_model) 
    return model

def calc_r2(pred, test_y, weights):
    """
    Calculates accuracy of predictions
    Arguments:
        pred - dataframe with predicted values
        test_y - dataframe with expected values
        weights - list with weights, each principal component has different weight based on its importance
            it is best to use previously calculated explained variance ratio from pca.
    Returns:
        r2 - r2_score metric 
    """
    pred_list = list()
    test_y_list = list()

    for col in pred.columns:
        pred_list.append(pred[col].values)
    for col in test_y.columns:
        test_y_list.append(test_y[col].values)
    
    r2_list = list()
    r2 = 0
    for i in range(0, 4):
        r2_list.append(r2_score(test_y_list[i], pred_list[i]))

    for i in range(0,4):
        r2 = r2 + r2_list[i] * weights[i]

    return r2_list, r2

def define_ann():
    """
    Creates an ANN
    Returns:
        model - keras unfitted model
    """
    
    input_layer = Input(shape=(96*4, ))
    dense_11 = Dense(384, activation='tanh')(input_layer)
    dense_21 = Dense(384, activation='tanh')(input_layer)
    dense_31 = Dense(384, activation='tanh')(input_layer)
    dense_41 = Dense(384, activation='tanh')(input_layer)
    """
    dense_12 = Dense(400, activation='tanh')(dense_11)
    dense_22 = Dense(400, activation='tanh')(dense_21)
    dense_32 = Dense(400, activation='tanh')(dense_31)
    dense_42 = Dense(400, activation='tanh')(dense_41)
    """
    merge_layer = concatenate([dense_11, dense_21, dense_31, dense_41], axis=-1)
    dense_merged_1 = Dense(300, activation='tanh')(merge_layer)
    norm1 = BatchNormalization()(dense_merged_1)
    drop1 = Dropout(0.2)(norm1)
    dense_merged_2 = Dense(400, activation='tanh')(drop1)
    
    dense_13 = Dense(200, activation='tanh')(dense_merged_2)
    dense_23 = Dense(200, activation='tanh')(dense_merged_2)
    dense_33 = Dense(200, activation='tanh')(dense_merged_2)
    dense_43 = Dense(200, activation='tanh')(dense_merged_2)

    drop_1 = Dropout(0.2)(dense_13)
    drop_2 = Dropout(0.2)(dense_23)
    drop_3 = Dropout(0.2)(dense_33)
    drop_4 = Dropout(0.2)(dense_43)
    """
    dense_14 = Dense(150, activation='tanh')(drop_1)
    dense_24 = Dense(150, activation='tanh')(drop_2)
    dense_34 = Dense(150, activation='tanh')(drop_3)
    dense_44 = Dense(150, activation='tanh')(drop_4)
    """
    out_1 = Dense(96, activation='linear')(drop_1)
    out_2 = Dense(96, activation='linear')(drop_2)
    out_3 = Dense(96, activation='linear')(drop_3)
    out_4 = Dense(96, activation='linear')(drop_4)

    model = Model(inputs=[input_layer],
                  outputs=[out_1, out_2, out_3, out_4])
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1.e-9, decay=0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=opt)

    return model

def train_ann(model, train_x, train_y, test_x, test_y, epochs):
    """
    Trains the ann model
    Arguments:
        model - keras model
        train_x - training data x 
        train_y - training data labels
        epochs - number of epochs to train
    """

    history = model.fit(train_x, [train_y[0], train_y[1], train_y[2], train_y[3]], epochs=epochs, batch_size=16, validation_data=(test_x, [test_y[0], test_y[1], test_y[2], test_y[3]]), verbose=2, shuffle=False)
    plt.plot(history.history['loss'], label="train")
    plt.plot(history.history['val_loss'], label='test')

    plt.legend()
    plt.show()

    model_json = model.to_json()
    with open('model_ann.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('model_ann.h5')
    print('Model saved')

def reframe_ann(train_x, train_y, test_x, test_y):
    """
    Reframes dataframes to numpy arrays for keras model
    """
    tr_x = train_x.astype('float32').values
    te_x = test_x.astype('float32').values
  

    test_y_1 = np.empty((test_y[0].shape[0], 96))
    test_y_2 = np.empty((test_y[0].shape[0], 96))
    test_y_3 = np.empty((test_y[0].shape[0], 96))
    test_y_4 = np.empty((test_y[0].shape[0], 96))

  
    for i in range(0, test_y[0].shape[0]):
        k = test_y[0][i]
        test_y_1[i] = k[k.columns.values[0]].astype('float32').values
        test_y_1[i] = k[k.columns.values[1]].astype('float32').values
        test_y_1[i] = k[k.columns.values[2]].astype('float32').values
        test_y_1[i] = k[k.columns.values[3]].astype('float32').values
    
    te_y = [test_y_1, test_y_2, test_y_3, test_y_4]
    
    train_y_1 = np.empty((train_y[0].shape[0], 96))
    train_y_2 = np.empty((train_y[0].shape[0], 96))
    train_y_3 = np.empty((train_y[0].shape[0], 96))
    train_y_4 = np.empty((train_y[0].shape[0], 96))

    for i in range(0, train_y[0].shape[0]):
        k = train_y[0][i]
        train_y_1[i] = k[k.columns.values[0]].astype('float32').values
        train_y_1[i] = k[k.columns.values[1]].astype('float32').values
        train_y_1[i] = k[k.columns.values[2]].astype('float32').values
        train_y_1[i] = k[k.columns.values[3]].astype('float32').values
    
    tr_y = [train_y_1, train_y_2, train_y_3, train_y_4]

    return tr_x, tr_y, te_x, te_y