# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import supervised_learning
import exploring_dataset
import preprocess
import unsupervised_learning

if __name__ == '__main__':
    data = preprocess.processDS("diabetes_binary_health_indicators_BRFSS2015.csv")

    #unsupervised_learning.kMeans(data)
    unsupervised_learning.doDBSCAN(data)


    #exploring_dataset.hexbinPlot(data, 'BMI', 'Income')

    #supervised_learning.logisticRegression(data)




