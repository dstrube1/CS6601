from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.datasets import load_wine
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from submission import load_csv

## For visualizing hand_binary.csv 
features,classes = load_csv('./data/hand_binary.csv',-1)
feat_names = ['Col0','Col1', 'Col2', 'Col3']
class_names = ['0', '1']

# # Call sklearn decision tree classifier and fit the decision tree on the dataset.
estimator = DecisionTreeClassifier()
estimator.fit(features, classes)

# Call graph visualization library
graph = Source(tree.export_graphviz(estimator, out_file=None
   , feature_names=feat_names, class_names=class_names 
   , filled = True))
   
# Display Decision Tree
display(SVG(graph.pipe(format='svg')))

"""
Traceback (most recent call last):
  File "/opt/miniconda3/envs/ai_env/lib/python3.7/site-packages/graphviz/backend.py", line 170, in run
    proc = subprocess.Popen(cmd, startupinfo=get_startupinfo(), **kwargs)
  File "/opt/miniconda3/envs/ai_env/lib/python3.7/subprocess.py", line 800, in __init__
    restore_signals, start_new_session)
  File "/opt/miniconda3/envs/ai_env/lib/python3.7/subprocess.py", line 1551, in _execute_child
    raise child_exception_type(errno_num, err_msg, err_filename)
FileNotFoundError: [Errno 2] No such file or directory: 'dot': 'dot'

brew update

brew install python-graphviz
conda install python-graphviz
"""
