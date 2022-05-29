import pandas as pd
from ExtremeLearningMachine import ELM
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


'''Get diagonal and lower triangular pairs of correlation matrix'''
def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols          = df.columns
    
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            
            pairs_to_drop.add((cols[i], cols[j]))
            
    return pairs_to_drop


# Parser for one file
def parse_file(file_data):
    
    line_dfs = []
    no_lines = len(file_data)

    for idx in np.arange(no_lines):

        # Read line
        line        = file_data.iloc[idx]

        # Get feature name
        feature     = line.iloc[0]

        # Parse data line
        data        = line.iloc[1].replace('[','').split('],')
        data        = [tuple(map(float, s.replace(']','').split(','))) for s in data]

        # Make dataframe column names
        colnames = [feature + '_' + elem for elem in feature if isinstance(elem, str)]
        colnames = colnames[0:len(data[0])]

        # Make dataframe out of this line
        line = pd.DataFrame(data, columns = colnames)

        # Append to list
        line_dfs.append(line)

    # concatenate list of dfs
    line_dfs = pd.concat(line_dfs, axis = 1)
    
    return line_dfs

def make_sliding_window(X, group_var, window):
    ''' Function to include lags of the predictors as additional predictors for each experiment'''
    
    if window > 0:
        
        # Make dataframe from numpy array
        X_df = pd.DataFrame(X)
        X_df['group'] = group_var

        # List to hold shifted dataframes (one per experiment)
        shifted_data = []

        # Groupby experiment
        for group, data in X_df.groupby('group'):

            # Shift experiment dataframe
            new_df = pd.concat([data.shift(i) for i in range(window)], axis = 1)

            # Drop group column
            new_df.drop('group', axis = 1, inplace = True)

            # Append to list
            shifted_data.append(new_df) 

        # List of experiments to new dataframe
        new_X = pd.concat(shifted_data, axis = 0)

        # Convert NaNs on the first rows to zeroes
        new_X.fillna(value = 0, inplace = True)
        
    else:
        
        # Do not perform sliding window
        new_X = pd.DataFrame(X)
    
    return new_X.values

def preprocess(X_t, X_v, y_t, y_v):
    ''' Apply preprocessing pipeline '''
    
    # Scale
    scaler = MinMaxScaler(feature_range = (-1, 1))
    X_tn   = scaler.fit_transform(X_t)
    X_vn   = scaler.transform(X_v)

    # One-hot encode targets
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(y_t.reshape(-1, 1))
    
    y_t_oh = enc.transform(y_t.reshape(-1, 1)).toarray()
    y_v_oh = enc.transform(y_v.reshape(-1, 1)).toarray()
    
    return X_tn, X_vn, y_t_oh, y_v_oh


def train_predict(X, y, elm_hidden, t_idx, v_idx):
    '''' Train on the test and compute score on a validation set '''
    
    # Make training / validation sets for this fold
    X_t, y_t = X[t_idx, :], y[t_idx]
    X_v, y_v = X[v_idx, :], y[v_idx]

    # Apply preprocessing
    X_tn, X_vn, y_toh, y_voh = preprocess(X_t, X_v, y_t, y_v)

    # Train ELM
    elm = ELM(input_size = X_tn.shape[1], hidden_size = elm_hidden)
    elm.fit(X_tn, y_toh)

    # Predict
    y_hat = elm.predict(X_vn)

    # Grab AUC-ROC
    score = roc_auc_score(y_voh, y_hat, average = 'weighted', multi_class = 'ovr')
                
    return score