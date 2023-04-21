import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def load_data(file_path: str, split: bool = True):
    """
    Load data from file_path and return numpy data

    Parameters
    ----------
    file_path : str
        The path of input data file (tab separated).
    split : bool
        Whether or not to return test set.

    Returns
    ----------
    (X_train, y_train)
    
    : training numpy array if split = False, else
    (X_train, y_train), (X_test, y_test): training and testing numpy array if split = True
    """
    # YOUR CODE HERE
    df = pd.read_csv(file_path, sep="\t")

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=520)
        return X_train, y_train, X_test, y_test
    else:
        return X, y
    raise NotImplementedError()

# 4

# Decision Tree class
# You should implement the ID3 algorithm here
# You can add other utility methods to make your code easy to read :) 
# 4

# Decision Tree class
# You should implement the ID3 algorithm here
# You can add other utility methods to make your code easy to read :) 

class DecisionTree:
    def __init__(self):
        self.node = {}
        self.depth = 0

    @staticmethod
    def entropy(division):
        """ Returns the entropy of the dataset before splitting """
        n_samples = len(division)
        classes = set(division)
        entropy = 0
        for class_name in classes:
            prop = len([division[division == class_name]]) / n_samples
            entropy += prop * np.log2(prop)
        return -entropy

    @staticmethod
    def entropy_attribute(X_train, y_train, column_idx: int):
        """ Returns the weighted average entropy from the entropy """
        samples = X_train[:,column_idx]
        n_samples = len(samples)
        classes = set(samples)
        result = 0
        for class_name in classes:
            mask = samples == class_name
            y_sub = y_train[mask]
            entropy = DecisionTree.entropy(y_sub)
            result += (len(y_sub) / n_samples) * entropy
        return result

    @staticmethod
    def find_best_split(X_train, y_train):
        ''' Parameters:
        -------------
        X_train: Array of feature vectors
        y_train: Target vector
        -------------
        Returns the index of the column with the max infomation gain to split '''
        entropy = DecisionTree.entropy(y_train)
        maxGain, maxIdx = -1, -1
        for column_idx in range(len(X_train[0])):
            gain = entropy - DecisionTree.entropy_attribute(X_train, y_train, column_idx)
            if gain > maxGain:
                maxIdx = column_idx
                maxGain = gain
        return maxIdx

    def fit(self, X_train, y_train, val=None, col_idx=None):
        # 3
        # YOUR CODE HERE
        # Stop condition: All samples have the same label => Return leaf node
        if np.all(y_train == y_train[0]):
            leaf = {'val': y_train[0], 'col_idx': None, 'children': None}
            return {'val': val, 'col_idx': col_idx, 'children': [leaf]}

        column_idx = DecisionTree.find_best_split(X_train, y_train)
        classes = set(X_train[:,column_idx])
        node = {'val': val, 'col_idx': column_idx, 'children': []}
        for class_name in classes:
            mask = X_train[:,column_idx] == class_name
            X_sub = X_train[mask]
            y_sub = y_train[mask]
            node['children'].append(self.fit(X_sub, y_sub, class_name, column_idx))

        self.node = node
        self.depth += 1
      
        return node
        raise NotImplementedError()

    def predict(self, X_test):
        # 0.5
        # YOUR CODE HERE
        predict = ['' for _ in range(len(X_test))]
        for i, row in enumerate(X_test):
            predict[i] = self._predict(row)
        return predict
        raise NotImplementedError()

    def _predict(self, row):
        cur_layer = self.node
        while cur_layer['children'] is not None:
            for node in cur_layer['children']:
                if node['children'] is None: 
                    return node['val']
                if node['val'] == row[cur_layer['col_idx']]:
                    cur_layer = node
                    break

    def visualize(self, root, column_names, n_tabs=0):
        # 0.5
        # YOUR CODE HERE
        tab = '\t' * n_tabs
        for node in root['children']:
            if tab != '':
                for t in tab:
                    print('|' + t, end='')
            print(f"{column_names[root['col_idx']]} = {node['val']}", end='')
            if node['children'][0]['col_idx'] is None:
                print(f": {node['children'][0]['val']}")
            else:
                print('')
                self.visualize(node, column_names, n_tabs + 1)
        #raise NotImplementedError()
    
# 0.5 = 0.25 (tennis dataset) + 0.25 (titanic2 dataset) 

### NOTE: Flow to run your code (do this for all your datasets)

# dataset 1 (create one cell for each dataset with the following content)

tree = DecisionTree()
X_train, y_train, X_test, y_test = load_data("data/tennis.txt")
tree.fit(X_train, y_train)
y_hat_train = tree.predict(X_train) 
acc_train = accuracy_score(y_train, y_hat_train)
print(f'Training accuracy: {acc_train}')
y_hat_test = tree.predict(X_test) 
acc_test = accuracy_score(y_test, y_hat_test)
print(f'Accuracy: {acc_test}')

df = pd.read_csv("data/tennis.txt", sep="\t")
column_names = np.array(df.columns)
tree.visualize(tree.node, column_names)