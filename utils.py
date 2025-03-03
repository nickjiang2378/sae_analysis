from tqdm import tqdm
import uuid
import numpy as np
import json
import pickle
import pandas as pd

from scipy.sparse import csr_matrix

SAMPLE_TRUNCATION_LENGTH = 100

class DatasetFeatureActivations():
    def __init__(self, data, client = None, variant = None, dataset_description = ""):
        """
        Initialize a DatasetFeatureActivations instance.

        :param data: List of data samples
        :param client: Goodfire client
        :param variant: Goodfire variant (8b or 70b)
        :param dataset_description: Optional description of the dataset
        """
        assert isinstance(data, list), "data must be a list"
        self.data = data
        self.id = str(uuid.uuid4())
        self.dataset_description = dataset_description
        self._feature_activations = []
        for sample in tqdm(self.data, desc = "Calculating feature activations"):
            self._feature_activations.append(FeatureActivation(sample, client, variant))

        print(f"Initialized dataset feature activations {self.id}")

    def save_to_file(self, file_path):
        """
        Save the DatasetFeatureActivations instance to a file.

        :param file_path: Path to the file where the instance will be saved
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    def top_samples_for_feature(self, feature_index, top_k = 10, feature_activation_type = "max"):
        """
        Retrieve the top samples for a specific feature based on activation type.

        :param feature_index: Index of the feature
        :param top_k: Number of top samples to retrieve
        :param feature_activation_type: Type of feature activation ('max' or 'mean')
        :return: List of top samples and their activations
        """
        feature_activations = self.feature_activations(feature_activation_type)
        return list(zip(*sorted(zip(self.data, feature_activations[:, feature_index]), key = lambda x: x[1], reverse = True)[:top_k]))

    def feature_activations(self, feature_activation_type = "max"):
        """
        Get the feature activations for all samples.

        :param feature_activation_type: Type of feature activation ('max', 'mean', or 'sum')
        :return: Numpy array of feature activations
        """
        if feature_activation_type == "max":
            return np.concatenate([feature_activation.get_max_activations() for feature_activation in self._feature_activations])
        elif feature_activation_type == "mean":
            return np.concatenate([feature_activation.get_mean_activations() for feature_activation in self._feature_activations])
        elif feature_activation_type == "sum":
            return np.concatenate([feature_activation.get_sum_activations() for feature_activation in self._feature_activations])
        else:
            raise ValueError(f"Unsupported feature activation type: {feature_activation_type}")

    def _get_feature_labels(self):
        """
        Retrieve feature labels from a JSON file.

        :return: List of feature labels as strings
        """
        feature_labels_as_dict = json.load(open("feature_labels.json", "r"))
        feature_labels_as_str = []
        max_feature_index = max([int(i) for i in feature_labels_as_dict.keys()])
        for i in range(max_feature_index + 1):
            if str(i) in feature_labels_as_dict:
                feature_labels_as_str.append(feature_labels_as_dict[str(i)])
            else:
                feature_labels_as_str.append(f"feature_{i}")
        return feature_labels_as_str

    def diff(self, other, feature_activation_type = "max"):
        """
        Calculate the difference in feature activations between two datasets.

        :param other: Another DatasetFeatureActivations instance
        :param feature_activation_type: Type of feature activation ('max', 'mean', or 'sum')
        :return: List of tuples containing feature labels and their differences
        """
        if not isinstance(other, DatasetFeatureActivations):
            raise ValueError("Unsupported operand type(s) for -: 'DatasetFeatureActivations' and '{}'".format(type(other)))

        diff_feature_activations = np.mean(self.feature_activations(feature_activation_type), axis = 0) - np.mean(other.feature_activations(feature_activation_type), axis = 0)

        feature_labels = self._get_feature_labels()
        num_features = len(feature_labels)
        return pd.DataFrame({
            'feature': feature_labels,
            'diff_activation': diff_feature_activations[:num_features]
        }).sort_values(by = "diff_activation", ascending = False, ignore_index = True)

    def sort_by_features(self, features, feature_activation_type = "max"):
        """
        Sort data samples by specified features and activation type.

        :param features: List of feature indices to sort by
        :param feature_activation_type: Type of feature activation ('max', 'mean', or 'sum')
        :return: List of sorted data samples
        """
        feature_activations = self.feature_activations(feature_activation_type)

        selected_feature_activations = feature_activations[:, features]
        feature_labels = self._get_feature_labels()
        top_features = []
        for activation in selected_feature_activations:
            top_feature_index = next((i for i, val in enumerate(activation) if val != 0), -1)
            if top_feature_index != -1:
                top_features.append((activation[top_feature_index], feature_labels[features[top_feature_index]]))
            else:
                top_features.append((0, "NA"))
        sorted_data = sorted(
            [(data, feature[0], feature[1]) for data, feature in zip(self.data, top_features)],
            key=lambda x: x[1],
            reverse=True
        )
        return pd.DataFrame(sorted_data, columns=['sample', 'feature activation', 'top feature'])

    @staticmethod
    def load_from_file(file_path):
        """
        Load a DatasetFeatureActivations instance from a file.

        :param file_path: Path to the file
        :return: DatasetFeatureActivations instance
        """
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
            print(f"Loaded dataset {obj.id}{f': {obj.dataset_description}' if obj.dataset_description else ''}")
            return obj

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __next__(self):
        return next(self.data)


class FeatureActivation():
    def __init__(self, text_sample, client = None, variant = None, mean_activations = None, max_activations = None, sum_activations = None):
        """
        Initialize a FeatureActivation instance.

        :param text_sample: Text sample for which activations are calculated
        :param client: Goodfire client
        :param variant: Goodfire variant (8b or 70b)
        :param mean_activations: Optional precomputed mean activations
        :param max_activations: Optional precomputed max activations
        :param sum_activations: Optional precomputed sum activations
        """
        self.text_sample = text_sample

        if mean_activations is None or max_activations is None or sum_activations is None:
            if client is None or variant is None:
                raise ValueError("Client and variant must be provided if mean_activations, max_activations, and sum_activations are not provided")
            matrix = client.features.activations(
              messages=[{"role": "user", "content": text_sample}],
              model=variant,
            )
            self.mean_activations = csr_matrix(matrix.mean(axis = 0)) # most feature activations are 0
            self.max_activations = csr_matrix(matrix.max(axis = 0))
            self.sum_activations = csr_matrix(matrix.sum(axis = 0))
        else:
            self.mean_activations = mean_activations
            self.max_activations = max_activations
            self.sum_activations = sum_activations

    def get_mean_activations(self):
        """
        Get the mean activations as a dense array.

        :return: Numpy array of mean activations
        """
        return self.mean_activations.toarray()

    def get_max_activations(self):
        """
        Get the max activations as a dense array.

        :return: Numpy array of max activations
        """
        return self.max_activations.toarray()

    def get_sum_activations(self):
        """
        Get the sum activations as a dense array.

        :return: Numpy array of sum activations
        """
        return self.sum_activations.toarray()

    def __repr__(self):
        return f"FeatureActivation('{self.text_sample[:SAMPLE_TRUNCATION_LENGTH] + '...' if len(self.text_sample) > SAMPLE_TRUNCATION_LENGTH else self.text_sample}')"

    def __add__(self, other):
        if isinstance(other, FeatureActivation):
            return FeatureActivation(self.text_sample, mean_activations = self.mean_activations + other.mean_activations, max_activations = self.max_activations + other.max_activations, sum_activations = self.sum_activations + other.sum_activations)
        else:
            raise ValueError("Unsupported operand type(s) for +: 'FeatureActivation' and '{}'".format(type(other)))

    def __sub__(self, other):
        if isinstance(other, FeatureActivation):
            return FeatureActivation(self.text_sample, mean_activations = self.mean_activations - other.mean_activations, max_activations = self.max_activations - other.max_activations, sum_activations = self.sum_activations - other.sum_activations)
        else:
            raise ValueError("Unsupported operand type(s) for -: 'FeatureActivation' and '{}'".format(type(other)))
