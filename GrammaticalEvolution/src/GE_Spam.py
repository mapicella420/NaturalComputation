#! /usr/bin/env python

# PonyGE2
# Copyright (c) 2017 Michael Fenton, James McDermott,
#                    David Fagan, Stefan Forstenlechner,
#                    and Erik Hemberg
# Hereby licensed under the GNU GPL v3.
""" Python GE implementation """
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys

from utilities.algorithm.general import check_python_version
from stats.stats import get_stats
from algorithm.parameters import params, set_params
from utilities.stats import trackers
from utilities.fitness.get_data import get_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

check_python_version()
matplotlib.use('tkagg')

def mane():
    """ Run program """
    set_params(sys.argv[1:])  # exclude the ponyge.py arg itself

    # Run evolution
    individuals = params['SEARCH_LOOP']()

    # Print final review
    get_stats(individuals, end=True)

    # reload the dataset
    X_training, y_training, X_test, y_test = get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])

    # Get the best individual
    y_training_prev = trackers.best_ever.eval_train
    y_prev  = trackers.best_ever.eval_test

    cm = confusion_matrix(y_test, y_prev, labels=[0, 1])
    print("Confusion Matrix:")
    print(cm)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Spam", "Spam"])
    # display matrix
    cm_display.plot()
    plt.show()


if __name__ == "__main__":
    mane()
