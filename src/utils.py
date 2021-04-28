import os
from copy import deepcopy

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from pgmpy.estimators import BicScore, BDeuScore, HillClimbSearch, K2Score, MmhcEstimator, PC
from pgmpy.models import BayesianModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from model import BayesianNetworkModel
# notebooks
# from .model import BayesianNetworkModel


def load_data(data_dir='../data', data_file='cmc.data'):
    '''
    Load CMC data to dataframe and set column names.
    '''
    data_path = os.path.join(data_dir, data_file)

    data = pd.read_csv(data_path, header=None)
    # setting column names
    data.columns = [
        'wife_age', 'wife_edu', 'husband_edu', 'n_children', 'wife_religion', 
        'wife_working', 'husband_occup', 'sol_index', 'media_exposure', 
        'class'
    ]

    return data


def split_data(data):
    '''
    Split data to train and test set (stratify on y).
    Return dict: {'train':{'X', 'y'}, 'test':{'X', 'y'}}
    '''
    X = data.drop('class', axis=1)
    y = data[['class']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y, 
        random_state=42
    )

    return {
        'train': {'X': X_train, 'y': y_train},
        'test': {'X': X_test, 'y': y_test},
    }


def discretize_data(data, continuous_attrs, n_bins=10):
    '''
    Discretize continous attributes with chosen bin number.
    '''
    _dataset = deepcopy(data)

    X_train = _dataset['train']['X']
    X_test = _dataset['test']['X']

    est = KBinsDiscretizer(
        n_bins=n_bins, 
        encode='ordinal',  # Return the bin identifier encoded as an integer value
        strategy='uniform'  # All bins in each feature have identical widths
    )
    # for attr in cont_attrs:
    #     values = data[attr].to_numpy()
    #     values = values.reshape((len(values), 1))
    #     data[attr+'_discrete'] = kbins.fit_transform(values)
    est.fit(X_train[continuous_attrs])
    X_train[continuous_attrs] = est.transform(X_train[continuous_attrs])
    X_test[continuous_attrs] = est.transform(X_test[continuous_attrs])

    _dataset['train']['X'] = X_train
    _dataset['test']['X'] = X_test

    return _dataset


def get_classification_report(y_true, y_pred):
    '''
    Return classification_report from sklearn.
    '''
    return classification_report(
        y_true=y_true,
        y_pred=y_pred
    )


def get_metrics(y_true, y_pred, average='micro', prec=3):
    '''
    Return accuracy, precision, recall and f1_score
        for given y_true and y_pred.
    '''
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)

    return {
        'accuracy': round(accuracy, prec),
        'precision': round(precision, prec),
        'recall': round(recall, prec),
        'f1': round(f1, prec)
    }


def plot_network(bn):
    '''
    Plot bayesian network.
    '''
    plt.figure(figsize=(10, 10))
    pos=graphviz_layout(bn, prog='dot')
    nx.draw(
        bn,
        pos=pos,
        with_labels=True,
        node_color='white',
        edgecolors='black',
        node_size=8000,
        arrowsize=20,
    )
    plt.show()


def run_experiment(data, network, estimator='BayesianEstimator'):
    # Declare data sets to variables
    X_train = data['train']['X']
    y_train = data['train']['y']
    X_test = data['test']['X']
    y_test = data['test']['y']
    train_ds = pd.concat([X_train, y_train], axis=1)
    test_ds = pd.concat([X_test, y_test], axis=1)

    nodes = list(train_ds.columns)

    bn_model = BayesianNetworkModel(
        nodes=nodes,
        network=network,
        fit_estimator=estimator
    )
    bn_model.fit(
        training_data=data['train']
    )
    y_pred = bn_model.predict(X_test)

    return get_metrics(
        y_true=y_test,
        y_pred=y_pred,
        average='macro'
    )
