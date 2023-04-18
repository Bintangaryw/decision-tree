from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
features = iris['data']
target = iris['target']
#print(features)
len(features)

decisiontree = DecisionTreeClassifier(random_state = 0, max_depth = None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0, max_leaf_nodes=None, min_impurity_decrease=0)

model = decisiontree.fit(features,target)
observation = [[5,4,3,2]]
model.predict(observation)
model.predict_proba(observation)

import pydotplus
from sklearn import tree
dot_data = tree.export_graphviz(decisiontree, out_file=None, features_names=iris['feature_names'], class_names=iris['target_names'])
from IPython.display import Image
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('iris.png')

