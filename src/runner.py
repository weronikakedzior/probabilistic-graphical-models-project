# from copy import deepcopy
import time

# import networkx as nx
import pandas as pd
# from matplotlib import pyplot as plt
# from networkx.drawing.nx_agraph import graphviz_layout
from pgmpy.estimators import BicScore, BDeuScore, HillClimbSearch, K2Score, MmhcEstimator, PC
from pgmpy.models import BayesianModel
from tqdm import tqdm

import utils
from model import BayesianNetworkModel
# notebooks:
# import src.utils as utils
# from src.model import BayesianNetworkModel


def run_experiments(
    discretized_data_dict: dict, 
    networks_dict: dict, 
    estimators_dict: dict,
    y_on='class'
):
    n_experiments = len(discretized_data_dict) * len(networks_dict) * len(estimators_dict)
    experiment_num = 0
    result_df = pd.DataFrame(columns=['network', 'n_bins', 'estimator', 
                                      'accuracy', 'precision', 'recall', 'f1', 'time'])

    for discretized_data_label, discretized_data in discretized_data_dict.items():
        for network_label, network in networks_dict.items():
            for estimator_label, estimator in estimators_dict.items():                    
                experiment_num += 1
                print(f'Processing experiment: {experiment_num} out of {n_experiments} ...', end=' ')
                start_time = time.time()

                model_results = utils.run_experiment(
                    data=discretized_data, 
                    network=network,
                    estimator=estimator, 
                    y_on=y_on
                )

                result_df = result_df.append({
                    'network': network_label, 
                    'n_bins': discretized_data_label, 
                    'estimator': estimator_label, 
                    'accuracy': model_results['accuracy'], 
                    'precision': model_results['precision'], 
                    'recall': model_results['recall'], 
                    'f1': model_results['f1'], 
                    'time': round(time.time()-start_time, 3)
                }, ignore_index=True)

                print('Done!')
    
    return result_df