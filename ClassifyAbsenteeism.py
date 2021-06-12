import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

input_file = "~/Documents/Harrisburg University/ANLY - 530/Project/Absenteeism_at_work_train.csv"

class model:
    def __init__(self, datafile = input_file):
        self.df = pd.read_csv(input_file)

if __name__ == '__main__':
    model_instance = Model()

if __name__ == '__main__':
    model_instance = Model()
    print(model_instance.df.head())

class model:
    def __init__(self, datafile = input_file):
        self.df = pd.read_csv(input_file)


