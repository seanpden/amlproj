# IMPORTS
# -------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score

from sklearn import tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
# -------------------------------------------------------------------


# DATA PREP - NAIVE BAYES
# -------------------------------------------------------------------

# Hardcoded vars for unique dataset
data = pd.read_csv("AMLdata.csv")

# All independent vars in data
dataFeatures = pd.DataFrame(data, columns=['Events | Median (HLA DR-FITC)',
       'Events | Median (CD117-PE)', 'Events | Median (CD45-ECD)',
       'Events | Median (CD34-PC5)', 'Events | Median (CD38-PC7)',
       'Events | Median (FS Lin)', 'Events | Median (SS Log)'])

# All dependent vars in data
dataLabels = pd.DataFrame(data, columns=['AML?'])

# debug print
# print("dataFeatures: ", dataFeatures)
# print("dataLabels: ", dataLabels)
# -------------------------------------------------------------------

# DATA PREP - DECISION TREE (many refs to naive bayes data prep)
# -------------------------------------------------------------------
dt_x = dataFeatures
dt_y = dataLabels
# -------------------------------------------------------------------

# DATA PREP - DECISION TREE (many refs to naive bayes data prep)
# -------------------------------------------------------------------
frt_x = dataFeatures
frt_y = dataLabels
# -------------------------------------------------------------------

# FUNCTIONS
# -------------------------------------------------------------------
def gauss_naive_bayes(test_size=0.25):
    '''
    Perform Gaussian Naive Bayes algorithm with the inputed 'test_size',
    prints and returns the metrics.accuracy_score.

    PARAMETERS:
    test_size: float or int, default = 0.25 \n
    If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. \n
    If int, represents the absolute number of test samples.
    '''
    n = test_size
    X_train, X_test, y_train, y_test = train_test_split(dataFeatures, dataLabels, test_size=n, random_state=53)
    
    gnb = GaussianNB()
    
    gnb.fit(X_train, y_train.values.ravel())
    
    y_pred = gnb.predict(X_test)
    
    print("Gaussian Naive Bayes accuracy_score: ", metrics.accuracy_score(y_test, y_pred))
    print("Gaussian Naive Bayes CrossValidate: ", gnb.score(X_test, y_test))
    print("Gaussian Naive Bayes F1 Score: ", f1_score(y_test, y_pred))
    print(y_test)
    print(y_pred)
    return metrics.accuracy_score(y_test, y_pred)

def decision_tree():
    '''
    Plots and calculates a DecisionTreeClassifier
    '''
    plt.figure(dpi=200)
    clf = tree.DecisionTreeClassifier().fit(dt_x, dt_y)
    plot_tree(clf, filled=True)
    plt.title("Decision tree trained on all features")
    plt.savefig("DecisionTreeOutput")
    plt.show()

def forest_random_tree(test_size=0.25):
    '''
    Performs the RandomForest algorithm with the inputed 'test_size',
    prints and returns the metrics.accuracy_score.

    PARAMETERS:
    test_size: float or int, default = 0.25 \n
    If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. \n
    If int, represents the absolute number of test samples.
    '''
    n = test_size
    X_train, X_test, y_train, y_test = train_test_split(frt_x, frt_y, test_size=n, random_state=53)

    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)
    
    print("RandomForest accuracy_score: ", metrics.accuracy_score(y_test, y_pred))
    print("RandomForest CrossValidate: ", clf.score(X_test, y_test))
    print("RandomForest F1 Score: ", f1_score(y_test, y_pred))
    # print(y_test)
    # print(y_pred)
    return metrics.accuracy_score(y_test, y_pred)
# -------------------------------------------------------------------


# FUNCTION CALLS
# -------------------------------------------------------------------
gauss_naive_bayes()
decision_tree()
forest_random_tree()
# -------------------------------------------------------------------