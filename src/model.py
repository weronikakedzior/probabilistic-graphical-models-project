import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.models import BayesianModel

# from hidden_prints import HiddenPrints, RedirectOutput, Capturing
from IPython.utils import io


class BayesianNetworkModel:
    def __init__(self, nodes: list, network, fit_estimator='BayesianEstimator'):
        self._nodes = nodes
        self._network = network
        self._fit_estimator = fit_estimator
        self._model = None
        self.build_model()
    

    def build_model(self):
        self._model = BayesianModel()
        self._model.add_nodes_from(self._nodes)
        self._model.add_edges_from(list(self._network.edges()))

    
    def get_model(self):
        return self._model
    

    def get_network(self):
        return self._network

    
    def fit(self, training_data: dict):
        '''
        Fit model to training data.
        '''
        X_train = training_data['X']
        y_train = training_data['y']

        train_ds = pd.concat([X_train, y_train], axis=1) 

        if self._fit_estimator == 'BayesianEstimator':
            self._model.fit(
                data=train_ds, 
                estimator=BayesianEstimator,
                prior_type="BDeu",
                equivalent_sample_size=10,
                complete_samples_only=False
            )
        elif self._fit_estimator == 'MaximumLikelihoodEstimator':
            self._model.fit(
                data=train_ds, 
                estimator=MaximumLikelihoodEstimator,
            )
        else:
            print('That estimator is not available!')
        
        # self._model.fit(
        #     data=train_ds,
        #     estimator=BayesianEstimator, 
        #     prior_type='dirichlet', 
        #     pseudo_counts=2
        # )


    def predict(self, X):    
        '''
        Make prediction on model.
        '''
        y_pred = []

        ve_infer = VariableElimination(self._model)
        # bp_infer = BeliefPropagation(model)

        with io.capture_output() as captured:
            for _, row in X.iterrows():
                evidence = dict(row)
                y_pred.append(
                    ve_infer.map_query(
                        variables=['class'], 
                        evidence=evidence
                    )['class']
                )

        return y_pred