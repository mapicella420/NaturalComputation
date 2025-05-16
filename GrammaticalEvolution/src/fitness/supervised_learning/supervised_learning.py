import numpy as np
import os
import pandas as pd

np.seterr(all="raise")

from algorithm.parameters import params
from utilities.fitness.get_data import get_data
from utilities.fitness.math_functions import *
from utilities.fitness.optimize_constants import optimize_constants

from fitness.base_ff_classes.base_ff import base_ff


class supervised_learning(base_ff):
    """
    Fitness function for supervised learning, ie regression and
    classification problems. Given a set of training or test data,
    returns the error between y (true labels) and yhat (estimated
    labels).

    We can pass in the error metric and the dataset via the params
    dictionary. Of error metrics, eg RMSE is suitable for regression,
    while F1-score, hinge-loss and others are suitable for
    classification.

    This is an abstract class which exists just to be subclassed:
    should not be instantiated.
    """

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        # Get training and test data
        self.training_in, self.training_exp, self.test_in, self.test_exp = \
            get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])

        # Find number of variables.
        self.n_vars = np.shape(self.training_in)[1] # sklearn convention

        # Regression/classification-style problems use training and test data.
        if params['DATASET_TEST']:
            self.training_test = True

    def evaluate(self, ind, **kwargs):
        dist = kwargs.get('dist', 'training')

        if dist == "training":
            x = self.training_in
            y = self.training_exp
            print("[DEBUG] Using TRAINING data")
        elif dist == "test":
            x = self.test_in
            y = self.test_exp
            print("[DEBUG] Using TEST data")
        else:
            raise ValueError("Unknown dist: " + dist)

        shape_mismatch_txt = """Shape mismatch between y and yhat. Please check
that your grammar uses the `x[:, 0]` style, not `x[0]`. See issue #130."""

        if params['OPTIMIZE_CONSTANTS']:
            if dist == "training":
                return optimize_constants(x, y, ind)
            else:
                phen = ind.phenotype_consec_consts
                c = ind.opt_consts
                print(f"[DEBUG] Test eval with optimized constants: {c}")
                yhat = eval(phen)
                if np.ndim(yhat) != 0 and y.shape != yhat.shape:
                    raise ValueError(shape_mismatch_txt)
                return params['ERROR_METRIC'](y, yhat)

        else:
            base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            dataset_path = os.path.normpath(os.path.join(base_path, "datasets", params['DATASET_TRAIN']))
            df = pd.read_csv(dataset_path)
            feature_names = list(df.columns[:-1])

            yhat = []
            print(f"[DEBUG] Individual phenotype: {ind.phenotype}")

            for i in range(len(x)):
                local_vars = {feature_names[j]: x[i][j] for j in range(len(feature_names))}
                try:
                    phen = ind.phenotype.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
                    result = eval(phen, {}, local_vars)
                    yhat.append(int(bool(result)))
                except Exception as e:
                    print(f"[ERROR] Eval failed at row {i}")
                    print(f"[ERROR] Phenotype: {ind.phenotype}")
                    print(f"[ERROR] Local vars: {local_vars}")
                    print(f"[ERROR] Exception: {e}")
                    yhat.append(0)

            yhat = np.array(yhat)
            print(f"[DEBUG] yhat prediction (first 10): {yhat[:10]}")
            print(f"[DEBUG] y true (first 10): {y[:10]}")

            if dist == 'training':
                ind.eval_train = yhat
            elif dist == 'test':
                ind.eval_test = yhat

            if np.ndim(yhat) != 0 and y.shape != yhat.shape:
                print(f"[ERROR] Shape mismatch: y shape = {y.shape}, yhat shape = {yhat.shape}")
                raise ValueError(shape_mismatch_txt)

            score = params['ERROR_METRIC'](y, yhat)
            print(f"[DEBUG] Score = {score}")
            return score

