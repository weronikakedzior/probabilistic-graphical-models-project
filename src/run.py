import os
from copy import deepcopy

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from pgmpy.estimators import BicScore, BDeuScore, HillClimbSearch, K2Score, MmhcEstimator, PC
from pgmpy.models import BayesianModel

import utils
from model import BayesianNetworkModel
from runner import run_experiments


def run(results_file='../out/xyz.csv', y_on='class'):
    # Data loading and train/test split
    data = utils.load_data()
    data = utils.split_data(data, y_on=y_on)

    # Data copy (without discretization)
    data_copy = deepcopy(data)

    continuous_attrs = ['wife_age', 'n_children']

    # Declare discretized data dict
    discretized_data_dict = {}
    for n_bins in range(3,10,3):  # [3,6,9]
        deta = data_copy
        key = str(n_bins)+'_bins'
        data = utils.discretize_data(
            data=data, 
            continuous_attrs=continuous_attrs,
            n_bins=n_bins,
            y_on=y_on
        )
        discretized_data_dict[key] = data

    data = discretized_data_dict['6_bins']
    X_train = data['train']['X']
    y_train = data['train']['y']
    X_test = data['test']['X']
    y_test = data['test']['y']
    train_ds = pd.concat([X_train, y_train], axis=1)
    test_ds = pd.concat([X_test, y_test], axis=1)
    nodes = list(train_ds.columns)
    # [
    #     'wife_age', 'wife_edu', 'husband_edu', 'n_children', 'wife_religion', 
    #     'wife_working', 'husband_occup', 'sol_index', 'media_exposure', 
    #     'class'
    # ]

    # Declare estimators dict
    estimators_dict = {
        'BayesianEstimator': 'BayesianEstimator', 
        'MaxLLEstimator': 'MaximumLikelihoodEstimator'
    }

    # Declare networks
    network_1 = BayesianModel([
        ('wife_age','n_children'), ('wife_age','wife_working'), ('wife_age','class'), 
        ('wife_edu','wife_working'), ('wife_edu','sol_index'), ('wife_edu','class'), 
        ('husband_edu','husband_occup'), ('husband_edu','sol_index'), ('husband_edu','class'), 
        ('n_children','class'), 
        ('wife_religion','class'), 
        ('wife_working','n_children'), ('wife_working','sol_index'), 
        ('husband_occup','n_children'), ('husband_occup','sol_index'), 
        ('sol_index','media_exposure'), 
        ('media_exposure','class'), 
    ])

    network_2 = BayesianModel([
        ('wife_age','n_children'), ('wife_age','wife_working'), 
        ('wife_edu','wife_working'), ('wife_edu','sol_index'), ('wife_edu','media_exposure'), ('wife_edu','class'),
        ('husband_edu','husband_occup'), ('husband_edu','sol_index'), ('husband_edu','media_exposure'), ('husband_edu','class'),
        ('n_children','wife_working'), ('n_children','husband_occup'), ('n_children','class'), 
        ('wife_religion','class'), 
        ('wife_working','sol_index'), 
        ('husband_occup','sol_index'), 
        ('sol_index','media_exposure'),
    ])

    network_3 = BayesianModel([
        ('wife_age','n_children'), ('wife_age','wife_working'), 
        ('wife_edu','wife_working'), ('wife_edu','class'),
        ('husband_edu','husband_occup'), ('husband_edu','class'),
        ('n_children','wife_working'), ('n_children','husband_occup'), ('n_children','class'), 
        ('wife_religion','class'), 
    ])
    network_3.add_node('sol_index')
    network_3.add_node('media_exposure')

    network_4 = BayesianModel([
        ('wife_age','n_children'), 
        ('wife_edu','class'),
        ('husband_edu','class'),
        ('n_children','class'), 
        ('wife_religion','class'), 
    ])
    network_4.add_node('sol_index')
    network_4.add_node('media_exposure')
    network_4.add_node('wife_working')
    network_4.add_node('husband_occup')

    network_5 = BayesianModel()
    for col in list(X_train.columns):
        network_5.add_edge(y_on, col)

    est = HillClimbSearch(train_ds)
    hcs_bic = est.estimate(scoring_method=BicScore(train_ds))
    hcs_bdeu = est.estimate(scoring_method=BDeuScore(train_ds))
    hcs_k2 = est.estimate(scoring_method=K2Score(train_ds))

    own_network_proposals = [
        network_1, network_2, network_3, 
        network_4, network_5
    ]
    hc_own_networks = []
    for own_network in own_network_proposals:
        hc_own_networks.append(
            est.estimate(scoring_method=K2Score(train_ds), start_dag=own_network.copy())
        )

    est = PC(train_ds)
    pc_network = est.estimate()

    pc_est = PC(data=train_ds)
    skeleton, separating_sets = pc_est.build_skeleton(variant='parallel')
    hc = HillClimbSearch(
        data=train_ds, 
        scoring_method=BDeuScore(data=train_ds)
    )
    pchc_learned_model = hc.estimate(
        tabu_length=100, 
        white_list=skeleton.to_directed().edges()
    )

    # Declare networks dict
    networks_dict = {
        #'network_1': network_1, 
        'network_2': network_2, 
        #'network_3': network_3, 
        #'network_4': network_4, 
        #'naive_bayes': network_5, 
        #'hcs_bic': hcs_bic, 
        #'hcs_bdeu': hcs_bdeu, 
        #'hcs_k2': hcs_k2, 
        #'pc': pc_network, 
        #'hcs_network_1': hc_own_networks[0], 
        #'hcs_network_2': hc_own_networks[1], 
        #'hcs_network_3': hc_own_networks[2], 
        #'hcs_network_4': hc_own_networks[3], 
        #'hcs_naive_bayes': hc_own_networks[4], 
        #'hcs_pc': pchc_learned_model
    }

    result_df = run_experiments(
        discretized_data_dict=discretized_data_dict, 
        networks_dict=networks_dict, 
        estimators_dict=estimators_dict, 
        y_on=y_on
    )

    result_df.to_csv(results_file)



if __name__ == '__main__':
    out_dir = '../out'
    out_file = 'results_class.csv'
    results_file = os.path.join(out_dir, out_file)

    run(
        results_file=results_file,
        y_on='n_children'  # n_children class
    )
