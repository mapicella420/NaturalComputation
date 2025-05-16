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
import time
from utilities.algorithm.general import check_python_version
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

check_python_version()

from stats.stats import get_stats
from algorithm.parameters import params, set_params
from utilities.stats import trackers
from utilities.fitness.get_data import get_data
from ErrorGrids import parkes_error_grid
from synthetic_tests_lib import crosscorr
import sys

matplotlib.use('tkagg')

def mane():
    """ Run program """
    print("✅ Inizio esecuzione - caricamento parametri")
    set_params(sys.argv[1:])
    start_time = time.time()

    print("⚙️  Avvio evoluzione con Grammatical Evolution...")
    individuals = params['SEARCH_LOOP']()
    print("✅ Evoluzione completata")

    print("📊 Calcolo statistiche finali...")
    get_stats(individuals, end=True)

    print("📂 Ricarico il dataset per valutare il miglior individuo...")
    training_in, training_real, test_in, test_real = get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])

    print("🔍 Calcolo predizioni del miglior individuo...")
    training_pred = trackers.best_ever.eval_train
    test_pred = trackers.best_ever.eval_test

    ### Metriche di classificazione
    print("📏 Metriche sul test set...")
    acc = accuracy_score(test_real, test_pred)
    f1 = f1_score(test_real, test_pred)
    prec = precision_score(test_real, test_pred)
    rec = recall_score(test_real, test_pred)

    print(f"📌 Accuracy  : {acc:.4f}")
    print(f"📌 F1-Score  : {f1:.4f}")
    print(f"📌 Precision : {prec:.4f}")
    print(f"📌 Recall    : {rec:.4f}")

    ### Confusion Matrix
    print("📈 Confusion Matrix (Test set)...")
    cm = confusion_matrix(test_real, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Risk", "High Risk"])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix - Test Set")
    plt.savefig("confusion_matrix.png", dpi=150)
    print("✅ Salvata come 'confusion_matrix.png'")

    end_time = time.time()
    print("⏱️ Tempo totale di esecuzione: {:.2f} secondi".format(end_time - start_time))


if __name__ == "__main__":
    mane()
