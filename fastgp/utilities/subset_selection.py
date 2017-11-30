import numpy as np

from fastgp.algorithms import fast_evaluate


class SubsetSelectionArchive(object):
    def __init__(self, frequency, predictors, response, subset_size, expression_dict):
        self.expression_dict = expression_dict
        self.frequency = frequency
        self.predictors = predictors
        self.response = response
        self.subset_size = subset_size
        self.num_obs = len(predictors)

        selected_indices = np.random.choice(self.num_obs, self.subset_size, replace=False)
        self.training_subset = np.zeros(self.num_obs, np.bool)
        self.training_subset[selected_indices] = 1
        self.subset_predictors = self.predictors[self.training_subset, :]
        self.subset_response = self.response[self.training_subset]
        self.generation_counter = 0

    def update(self, population):
        raise NotImplementedError

    def set_difficulty(self, errors):
        pass

    def get_data_subset(self):
        return self.subset_predictors, self.subset_response

    def get_indices(self):
        return np.arange(self.num_obs)[self.training_subset]

    def save(self, log_file):
        pass


class RandomSubsetSelectionArchive(SubsetSelectionArchive):
    def __init__(self, frequency, predictors, response, subset_size, expression_dict):
        SubsetSelectionArchive.__init__(self, frequency, predictors, response, subset_size, expression_dict)

    def update(self, population):
        if self.generation_counter % self.frequency == 0:
            selected_indices = np.random.choice(self.num_obs, self.subset_size, replace=False)
            self.training_subset = np.zeros(self.num_obs, np.bool)
            self.training_subset[selected_indices] = 1
            self.subset_predictors = self.predictors[self.training_subset, :]
            self.subset_response = self.response[self.training_subset]
            self.expression_dict.clear()
        self.generation_counter += 1


def fast_numpy_evaluate_subset(ind, context, subset_selection_archive, get_node_semantics,
                               inner_evaluate_function=fast_evaluate.fast_numpy_evaluate,
                               error_function=None, expression_dict=None):
    predictors, response = subset_selection_archive.get_data_subset()
    root_semantics = inner_evaluate_function(ind, context, predictors, get_node_semantics, error_function=None,
                                             expression_dict=expression_dict)
    return error_function(root_semantics, response)
