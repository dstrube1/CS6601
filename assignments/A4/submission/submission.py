import numpy as np
import math
from collections import Counter
import time
from sklearn.metrics import confusion_matrix as confusion_matrix_o  # original


class DecisionNode:
    """Class to represent a node or leaves in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """
        Create a decision node with eval function to select between left and right node
        NOTE In this representation 'True' values for a decision take us to the left.
        This is arbitrary, but testing relies on this implementation.
        Args:
            left (DecisionNode): left child node
            right (DecisionNode): right child node
            decision_function (func): evaluation function to decide left or right
            class_label (value): label for leaf node
        """
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Determine recursively the class of an input array by testing a value
           against a feature's attributes values based on the decision function.

        Args:
            feature: (numpy array(value)): input vector for sample.

        Returns:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label
        elif self.decision_function(feature):
            return self.left.decide(feature)
        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice index for data labels.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as numpy array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if class_index == -1:
        classes = out[:, class_index]
        features = out[:, :class_index]
        return features, classes
    elif class_index == 0:
        classes = out[:, class_index]
        features = out[:, 1:]
        return features, classes
    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data contained in the ReadMe.
    It must be built fully starting from the root.
    
    Returns:
        The root node of the decision tree.
    """
    # Defaults
    left = None
    right = None
    decision_function = None

    # Leaf values
    class_label_0 = 0
    class_label_1 = 1
    class_label_2 = 2

    # Leaf nodes
    leaf_0 = DecisionNode(left, right, decision_function, class_label_0)
    leaf_1 = DecisionNode(left, right, decision_function, class_label_1)
    leaf_2 = DecisionNode(left, right, decision_function, class_label_2)

    decision_tree_root = DecisionNode(left, right, lambda feature: 0.8 < feature[0] < 2)

    root_left = DecisionNode(left, right, lambda feature: feature[1] > -1.7)
    decision_tree_root.left = root_left

    root_left_left = leaf_1
    decision_tree_root.left.left = root_left_left

    root_left_right = leaf_0
    decision_tree_root.left.right = root_left_right

    root_right = DecisionNode(left, right, lambda feature: feature[1] < 0)
    decision_tree_root.right = root_right

    root_right_left = leaf_2
    decision_tree_root.right.left = root_right_left

    root_right_right = DecisionNode(left, right, lambda feature: -1 < feature[2] < -0.7)
    decision_tree_root.right.right = root_right_right

    root_right_right_left = leaf_2
    decision_tree_root.right.right.left = root_right_right_left

    root_right_right_right = leaf_0
    decision_tree_root.right.right.right = root_right_right_right

    return decision_tree_root


def confusion_matrix(true_labels, classifier_output, n_classes=2):
    """Create a confusion matrix to measure classifier performance.
   
    Classifier output vs true labels, which is equal to:
    Predicted  vs  Actual Values.
    
    Output will sum multiclass performance in the example format:
    (Assume the labels are 0,1,2,...n)
                                     |Predicted|
                     
    |A|            0,            1,           2,       .....,      n
    |c|   0:  [[count(0,0),  count(0,1),  count(0,2),  .....,  count(0,n)],
    |t|   1:   [count(1,0),  count(1,1),  count(1,2),  .....,  count(1,n)],
    |u|   2:   [count(2,0),  count(2,1),  count(2,2),  .....,  count(2,n)],'
    |a|   .............,
    |l|   n:   [count(n,0),  count(n,1),  count(n,2),  .....,  count(n,n)]]
    
    'count' function is expressed as 'count(actual label, predicted label)'.
    
    For example, count (0,1) represents the total number of actual label 0 and the predicted label 1;
                 count (3,2) represents the total number of actual label 3 and the predicted label 2.           
    
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
    Returns:
        A two-dimensional array representing the confusion matrix.
    """
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0
    if n_classes == 2:
        for i in range(len(classifier_output)):
            current_output = classifier_output[i]
            current_true_label = true_labels[i]
            if current_output == 1 and current_true_label == 1:
                true_positives += 1
            elif current_output == 1 and current_true_label == 0:
                false_positives += 1
            elif current_output == 0 and current_true_label == 0:
                true_negatives += 1
            else:
                false_negatives += 1
        for i in range(n_classes):
            pass
        c_matrix = [[true_negatives, false_positives], [false_negatives, true_positives]]
    else:
        labels = []
        for i in range(n_classes):
            labels.append(str(i))
        c_matrix = confusion_matrix_o(true_labels, classifier_output)  # , labels=labels)
    return c_matrix


def precision(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the precision of a classifier compared to the correct values.
    In this assignment, precision for label n can be calculated by the formula:
        precision (n) = number of correctly classified label n / number of all predicted label n 
                      = count (n,n) / (count(0, n) + count(1,n) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of precision of each classifier output. 
        So if the classifier is (0,1,2,...,n), the output should be in the below format: 
        [precision (0), precision(1), precision(2), ... precision(n)].
    """

    c_matrix = confusion_matrix(classifier_output, true_labels, n_classes)
    result = c_matrix[0][0] / (c_matrix[0][0] + c_matrix[1][0])
    return result


def recall(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the recall of a classifier compared to the correct values.
    In this assignment, recall for label n can be calculated by the formula:
        recall (n) = number of correctly classified label n / number of all true label n 
                   = count (n,n) / (count(n, 0) + count(n,1) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of recall of each classifier output.
        So if the classifier is (0,1,2,...,n), the output should be in the below format: 
        [recall (0), recall (1), recall (2), ... recall (n)].
    """

    c_matrix = confusion_matrix(classifier_output, true_labels)
    result = c_matrix[0][0] / (c_matrix[0][0] + c_matrix[0][1])

    return result


def accuracy(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """Get the accuracy of a classifier compared to the correct values.
    Balanced Accuracy Weighted:
    -Balanced Accuracy: Sum of the ratios (accurate divided by sum of its row) divided by number of classes.
    -Balanced Accuracy Weighted: Balanced Accuracy with weighting added in the numerator and denominator.

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The accuracy of the classifier output.
    """
    c_matrix = confusion_matrix(classifier_output, true_labels)
    result = (c_matrix[0][0] + c_matrix[1][1]) / (c_matrix[0][0] + c_matrix[0][1] + c_matrix[1][0] + c_matrix[1][1])
    return result


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0, 1, 2, ...
    Returns:
        Floating point number representing the gini impurity.
    """
    if len(class_vector) == 0:
        return 0
    class_array = np.array(class_vector)
    unique, counts = np.unique(class_array, return_counts=True)
    counts_dict = Counter(dict(zip(unique, counts)))
    probs_zero = counts_dict[0] / sum(counts_dict.values())
    probs_one = counts_dict[1] / sum(counts_dict.values())
    return 1 - (probs_zero ** 2 + probs_one ** 2)


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0, 1, 2....
        current_classes (list(list(int): A list of lists where each list has
            0, 1, 2, ... values).
    Returns:
        Floating point number representing the gini gain.
    """
    current_impurities = []
    for i in range(len(current_classes)):
        current_impurities.append(gini_impurity(current_classes[i]))

    weighted_sum = (current_impurities[0] * len(current_classes[0])) + (current_impurities[1] * len(current_classes[1]))
    weighted_average_impurity = weighted_sum / (len(current_classes[0]) + len(current_classes[1]))
    prev_impurity = gini_impurity(previous_classes)

    return prev_impurity - weighted_average_impurity


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=22):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """
        num_features = features.shape[1]

        left = None
        right = None
        decision_function = None
        if np.all(classes == classes[0]):
            return DecisionNode(left, right, decision_function, classes[0])
        if depth == self.depth_limit:
            return DecisionNode(left, right, decision_function, np.bincount(classes.astype(int)).argmax())

        gini_gains = np.zeros((num_features, 1))
        best_splits = np.zeros((num_features, 1))
        for i in range(num_features):
            current_feature = features[:, i]
            possible_splits = self.get_split_points(current_feature, classes)
            best_split, gini_gain = self.find_best_split(possible_splits, current_feature, classes)
            gini_gains[i] = gini_gain
            best_splits[i] = best_split

        best_attribute_idx = gini_gains.argmax()
        best_attribute = features[:, best_attribute_idx]
        best_split = best_splits[best_attribute_idx]
        positive_classes = classes[best_attribute >= best_split]
        negative_classes = classes[best_attribute < best_split]

        left_features = features[best_attribute >= best_split, :]
        right_features = features[best_attribute < best_split, :]
        left_branch = self.__build_tree__(left_features, positive_classes, depth + 1)
        right_branch = self.__build_tree__(right_features, negative_classes, depth + 1)

        return DecisionNode(left_branch, right_branch, lambda attribute: attribute[best_attribute_idx] > best_split)

    def get_split_points(self, feature: np.ndarray, classes: np.ndarray):
        sorted_feature, sorted_classes = zip(*sorted(zip(feature, classes)))
        possible_splits = []
        for i in range(len(sorted_feature) - 1):
            if sorted_classes[i] != sorted_classes[i + 1]:
                possible_splits.append((sorted_feature[i] + sorted_feature[i + 1]) / 2)

        return possible_splits

    def find_best_split(self, splits, feature: np.ndarray, classes: np.ndarray):
        gini_gains = np.zeros((len(splits)))
        for i, split in enumerate(splits):
            positive_classes = classes[feature >= split]
            negative_classes = classes[feature < split]
            gini_gains[i] = gini_gain(classes, [positive_classes, negative_classes])

        best_split = gini_gains.argmax()
        return splits[best_split], gini_gains[best_split]

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        class_labels = []

        for index in range(features.shape[0]):
            class_labels.append(self.root.decide(features[index, :]))

        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """
    folds = []
    features = dataset[0]
    classes = dataset[1]
    rand_indices = np.arange(features.shape[0])
    np.random.shuffle(rand_indices)
    random_features = features[rand_indices]
    random_classes = classes[rand_indices]
    feature_subsets = np.array_split(random_features, k)
    class_subsets = np.array_split(random_classes, k)

    for i in range(len(feature_subsets)):
        test_set_features = feature_subsets[i]
        test_set_classes = class_subsets[i]
        test_set = (test_set_features, test_set_classes)

        temp_features = [x for j, x in enumerate(feature_subsets) if j != i]
        temp_classes = [x for j, x in enumerate(class_subsets) if j != i]
        train_set_features = np.vstack(temp_features)
        train_set_classes = np.concatenate(temp_classes)
        train_set = (train_set_features, train_set_classes)

        folds.append((train_set, test_set))

    return folds


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees=200, depth_limit=5, example_subsample_rate=.1,
                 attr_subsample_rate=.3):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        num_examples = features.shape[0]
        num_features = features.shape[1]
        for i in range(self.num_trees):
            drawn_examples = np.random.choice(np.arange(num_examples), int(num_examples * self.example_subsample_rate),
                                              replace=True)
            drawn_features = np.random.choice(np.arange(num_features), int(num_features * self.attr_subsample_rate),
                                              replace=False)
            decision_tree = DecisionTree(self.depth_limit)
            decision_tree.fit(features[np.ix_(drawn_examples, drawn_features)], classes[drawn_examples])
            self.trees.append((decision_tree, drawn_features))

    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        Returns:
            votes (list(int)): m votes for each element
        """
        votes = []

        for index in range(features.shape[0]):
            current_example = features[index, :]
            counter = Counter()
            for tree, drawn_features in self.trees:
                prediction = tree.root.decide(current_example[drawn_features])
                counter[prediction] += 1
            votes.append(counter.most_common(1)[0][0])

        return votes


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self, num_clf=0, depth_limit=0, example_subsample_rt=0.0,
                 attr_subsample_rt=0.0, max_boost_cycles=0):
        """Create a boosting class which uses decision trees.
        Initialize and/or add whatever parameters you may need here.
        Args:
             num_clf (int): fixed number of classifiers.
             depth_limit (int): max depth limit of tree.
             example_subsample_rt (float): percentage of example samples.
             attr_subsample_rt (float): percentage of attribute samples.
             max_boost_cycles : ???
        """
        self.num_clf = num_clf
        self.depth_limit = depth_limit
        self.example_subsample_rt = example_subsample_rt
        self.attr_subsample_rt = attr_subsample_rt
        self.max_boost_cycles = max_boost_cycles
        self.classifier = RandomForest(5, 5, .5, .5)

    def fit(self, features, classes):
        """Build the boosting functions classifiers.
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        self.classifier.fit(features, classes)

    def classify(self, features):
        """Classify a list of features.
        Predict the labels for each feature in features to its corresponding class
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """
        return self.classifier.classify(features)


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """
        """vectorized = np.zeros(data.shape)

        # TODO: finish this.

        return vectorized"""
        self_mult = np.multiply(data, data)
        return data + self_mult

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        sliced = data[0:100, :]
        _sum = sliced.sum(axis=1)
        idx = np.argmax(_sum)

        return _sum[idx], idx

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """
        unique_dict = {}
        flattened = data.flatten()
        for item in flattened:
            if item > 0:
                if item in unique_dict:
                    unique_dict[item] += 1
                else:
                    unique_dict[item] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """
        flat_data = data.flatten()
        positive_data = flat_data[flat_data > 0]
        unique, counts = np.unique(positive_data, return_counts=True)
        unique_dict = dict(zip(unique, counts))
        return unique_dict.items()

    def non_vectorized_glue(self, data, vector, dimension='c'):
        """Element wise array arithmetic with loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        if dimension == 'c' and len(vector) == data.shape[0]:
            non_vectorized = np.ones((data.shape[0],data.shape[1]+1), dtype=float)
            non_vectorized[:, -1] *= vector
        elif dimension == 'r' and len(vector) == data.shape[1]:
            non_vectorized = np.ones((data.shape[0]+1,data.shape[1]), dtype=float)
            non_vectorized[-1, :] *= vector
        else:
            raise ValueError('This parameter must be either c for column or r for row')
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row, col] = data[row, col]
        return non_vectorized

    def vectorized_glue(self, data, vector, dimension='c'):
        """Array arithmetic without loops.
        This function takes a multidimensional array and a vector, and then combines
        both of them into a new multidimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multidimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        vectorized = None
        # TODO
        return vectorized

    def non_vectorized_mask(self, data, threshold):
        """Element wise array evaluation with loops.
        This function takes a multidimensional array and then populates a new
        multidimensional array. If the value in data is below threshold it
        will be squared.
        Args:
            data: multidimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        non_vectorized = np.zeros_like(data, dtype=float)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                val = data[row, col]
                if val >= threshold:
                    non_vectorized[row, col] = val
                    continue
                non_vectorized[row, col] = val**2

        return non_vectorized

    def vectorized_mask(self, data, threshold):
        """Array evaluation without loops.
        This function takes a multidimensional array and then populates a new
        multidimensional array. If the value in data is below threshold it
        will be squared. You are required to use a binary mask for this problem
        Args:
            data: multidimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        vectorized = None
        # TODO
        return vectorized


def return_your_name():
    # return your name
    return 'David Strube'
