import warnings
import random
from copy import deepcopy
import math

import numpy as np

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr
from scipy.stats import skew

from fastgp.utilities.metrics import mean_squared_error
from fastgp.utilities.symbreg import numpy_protected_div_dividend, numpy_protected_sqrt, numpy_protected_log_one


class Statistics:

    def __init__(self):
        self.scores = []
        self.generations = []
        self.num_features = []
        self.index = 0

    def add(self, gen, score, num_features):
        self.generations.append(gen)
        self.scores.append(score)
        self.num_features.append(num_features)

    def __iter__(self):
        return self

    def next(self):
        self.index += 1
        if self.index > len(self.num_features):
            raise StopIteration
        return self.generations[self.index], self.scores[self.index], self.num_features[self.index]


class Feature:

    def __init__(self, value, string, infix_string, size=0, fitness=1, original_variable=False):
        self.value = value
        self.fitness = fitness
        self.string = string
        self.infix_string = infix_string
        self.size = size
        self.original_variable = original_variable

    def __str__(self):
        return self.string


class Operator:

    def __init__(self, operation, parity, string, infix, infix_name):
        self.operation = operation
        self.parity = parity
        self.string = string
        self.infix = infix
        self.infix_name = infix_name


def square(x):
    return np.power(x, 2)


def cube(x):
    return np.power(x, 3)


def is_huge(x):
    return x > np.finfo(np.float64).max / 100000


def numpy_safe_exp(x):
    with np.errstate(invalid='ignore'):
        result = np.exp(x)
        if isinstance(result, np.ndarray):
            result[np.isnan(x)] = 1
            result[np.isinf(x)] = 1
            result[is_huge(x)] = 1
        elif np.isinf(result):
            result = 1
        elif np.isnan(x):
            result = 1
        elif is_huge(x):
            result = 1
        return result


def generate_operator_map(ops):
    opmap = {}
    for o in ops:
        opmap[o.infix_name] = o
    return opmap


operators = [
    Operator(np.add, 2, '({0} + {1})', 'add({0},{1})', 'add'),
    Operator(np.subtract, 2, '({0} - {1})', 'sub({0},{1})', 'sub'),
    Operator(np.multiply, 2, '({0} * {1})', 'mul({0},{1})', 'mul'),
    Operator(numpy_protected_div_dividend, 2, '({0} / {1})', 'div({0},{1})', 'div'),
    # Operator(numpy_safe_exp, 1, 'exp({0})'),
    Operator(numpy_protected_log_one, 1, 'log({0})', 'log({0})', 'log'),
    Operator(square, 1, 'sqr({0})', 'sqr({0})', 'sqr'),
    Operator(numpy_protected_sqrt, 1, 'sqt({0})', 'sqt({0})', 'sqt'),
    Operator(cube, 1, 'cbe({0})', 'cbe({0})', 'cbe'),
    Operator(np.cbrt, 1, 'cbt({0})', 'cbt({0})', 'cbt'),
    Operator(None, None, None, None, 'mutate'),
    Operator(None, None, None, None, 'transition')

]
operators_map = generate_operator_map(operators)


def init(num_additions, feature_names, predictors, seed):
    random.seed(seed)
    np.random.seed(seed)
    if num_additions is None:
        num_additions = math.ceil(predictors.shape[1] / 3)
    if feature_names is None:
        feature_names = ['x' + str(x) for x in range(len(predictors))]
    return num_additions, feature_names


def init_features(feature_names, predictors, preserve_originals, range_operations, variable_type_indices):
    features = []
    for i, name in enumerate(feature_names):
        features.append(Feature(predictors[:, i], name, name, original_variable=preserve_originals))
    for _ in range(range_operations):
        features.append(RangeOperation(variable_type_indices, feature_names, predictors))
    return features


def get_basis(features):
    basis = np.zeros((features[0].value.shape[0], len(features)))
    for i, f in enumerate(features):
        basis[:, i] = features[i].value
    basis = np.nan_to_num(basis)
    scaler = StandardScaler()
    basis = scaler.fit_transform(basis)
    return basis, scaler


def get_model(basis, response, time_series_cv, splits):
    if time_series_cv:
        cv = TimeSeriesSplit(n_splits=splits)
    else:
        cv = KFold(n_splits=splits)
    model = ElasticNetCV(l1_ratio=1, selection='random', cv=cv)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(basis, response)
        _, coefs, _ = model.path(basis, response, l1_ration=model.l1_ratio_, alphas=model.alphas_)
    return model, coefs, model.mse_path_


def get_selected_features(num_additions, features, tournament_probability):
    selected_features = []
    for _ in range(num_additions):
        feature = tournament_selection(features, tournament_probability)
        selected_features.append(feature)
    return selected_features


def get_coefficient_fitness(coefs, mse_path, threshold, response_variance):
    mse = np.mean(mse_path, axis=1)
    r_squared = 1 - (mse / response_variance)
    binary_coefs = coefs > threshold
    return binary_coefs.dot(r_squared)


def rank_by_coefficient(features, coefs, mse_path, num_additions, threshold, response_variance,
                        verbose):
    fitness = get_coefficient_fitness(coefs, mse_path, threshold, response_variance)
    for i, f in enumerate(features):
        f.fitness = fitness[i]
    new_features = list(filter(lambda x: x.original_variable is True, features))
    possible_features = list(filter(lambda x: x.original_variable is False, features))
    possible_features.sort(key=lambda x: x.fitness, reverse=True)
    new_features.extend(possible_features[0:num_additions + 1])
    new_features.sort(key=lambda x: x.fitness, reverse=True)
    print('Top performing features:')
    for i in range(10):
        print(new_features[i].string + ' - ' + str(new_features[i].fitness))
    return new_features


def remove_zeroed_features(model, features, threshold, verbose):
    remove_features = []
    for i, coef in enumerate(model.coef_):
        features[i].fitness = math.fabs(coef)
        if features[i].fitness <= threshold and not features[i].original_variable:
            remove_features.append(features[i])
    for f in remove_features:
        features.remove(f)
    print('Removed ' + str(len(remove_features)) + ' features from population.')
    if verbose and remove_features:
        print(get_model_string(remove_features))
    return features


def update_fitness(features, response, threshold, fitness_algorithm, response_variance, num_additions,
                   time_series_cv, splits, verbose):
    basis, _ = get_basis(features)
    model, coefs, mse_path = get_model(basis, response, time_series_cv, splits)
    if fitness_algorithm == 'zero_out':
        features = remove_zeroed_features(model, features, threshold, verbose)
    elif fitness_algorithm == 'coefficient_rank':
        features = rank_by_coefficient(features, coefs, mse_path, num_additions, threshold, response_variance,
                                       verbose)
    return features


def uncorrelated(parents, new_feature, correlation_threshold):
    uncorr = True
    if type(parents) == list:
        for p in parents:
            r, _ = pearsonr(new_feature.value, p.value)
            if r > correlation_threshold:
                uncorr = False
    else:
        r, _ = pearsonr(new_feature.value, parents.value)
        if r > correlation_threshold:
            uncorr = False
    return uncorr


def tournament_selection(population, probability):
    individuals = random.choices(population, k=2)
    individuals.sort(reverse=True, key=lambda x: x.fitness)
    if random.random() < probability:
        return individuals[0]
    else:
        return individuals[1]


def compose_features(num_additions, features, tournament_probability, correlation_threshold,
                     range_operators, verbose):
    new_feature_list = []
    for _ in range(num_additions):
        operator = random.choice(operators)
        if operator.parity == 1:
            parent = tournament_selection(features, tournament_probability)
            new_feature_string = operator.string.format(parent.string)
            new_infix_string = operator.infix.format(parent.infix_string)
            new_feature_value = operator.operation(parent.value)
            new_feature = Feature(new_feature_value, new_feature_string, new_infix_string,
                                  size=parent.size + 1)
            if uncorrelated(parent, new_feature, correlation_threshold):
                new_feature_list.append(new_feature)
        elif operator.parity == 2:
            parent1 = tournament_selection(features, tournament_probability)
            parent2 = tournament_selection(features, tournament_probability)
            new_feature_string = operator.string.format(parent1.string, parent2.string)
            new_infix_string = operator.infix.format(parent1.infix_string, parent2.infix_string)
            new_feature_value = operator.operation(parent1.value, parent2.value)
            new_feature = Feature(new_feature_value, new_feature_string, new_infix_string,
                                  size=parent1.size + parent2.size + 1)
            if uncorrelated([parent1, parent2], new_feature, correlation_threshold):
                new_feature_list.append(new_feature)
        if range_operators:
            protected_range_operators = list(filter(lambda x: type(x) == RangeOperation and x.original_variable,
                                                    features))
            transitional_range_operators = list(filter(lambda x: type(x) == RangeOperation and not x.original_variable,
                                                       features))
            if operator.infix_name == 'transition' and protected_range_operators:
                parent = random.choice(protected_range_operators)
                new_feature = deepcopy(parent)
                new_feature.original_variable = False
                new_feature_list.append(new_feature)
            elif operator.infix_name == 'mutate' and transitional_range_operators:
                parent = random.choice(transitional_range_operators)
                new_feature = deepcopy(parent)
                new_feature.mutate_parameters()
                new_feature_list.append(new_feature)
    filtered_feature_list = list(filter(lambda x: x.size < 5, new_feature_list))
    features.extend(filtered_feature_list)
    print('Adding ' + str(len(filtered_feature_list)) + ' features to population.')
    if verbose:
        print(get_model_string(new_feature_list))
    return features


def score_model(features, response, time_series_cv, splits):
    print('Scoring model with ' + str(len(features)) + ' features.')
    basis, scaler = get_basis(features)
    model, _, _ = get_model(basis, response, time_series_cv, splits)
    score = mean_squared_error(model.predict(basis), response)[0]
    return score, model, scaler


def get_model_string(features):
    feature_strings = []
    for f in features:
        feature_strings.append(f.string)
    return '[' + '] + ['.join(feature_strings) + ']'


def compute_operation(num_variables, predictors, stack, feature_names):
    variables = []
    for _ in range(num_variables):
        variable_name = stack.pop()
        variable_index = feature_names.index(variable_name)
        variables.append(predictors[:, variable_index])
    operator = stack.pop()
    result = operator.operation(*variables)
    return result


def build_operation_stack(string):
    stack = []
    start = 0
    for i, s in enumerate(string):
        if s == '(':
            substring = string[start:i]
            start = i + 1
            stack.append(substring)
        elif s == ',':
            if i != start:
                substring = string[start:i]
                stack.append(substring)
            start = i + 1
        elif s == ')':
            if i != start:
                substring = string[start:i]
                stack.append(substring)
            start = i + 1
    return stack


def get_feature_value(stack, feature_names, predictors, variable_type_indices):
    variables_stack = []
    while len(stack) > 0:
        current = stack.pop()
        if variable_type_indices and current.startswith('RangeOperation'):
            range_operation = RangeOperation(variable_type_indices, feature_names, predictors, string=current)
            variables_stack.append(np.squeeze(range_operation.value))
        elif current in feature_names:
            variable_index = feature_names.index(current)
            variables_stack.append(predictors[:, variable_index])
        elif current in operators_map:
            operator = operators_map[current]
            variables = []
            for _ in range(operator.parity):
                variables.append(variables_stack.pop())
            result = operator.operation(*variables)
            variables_stack.append(result)
    return variables_stack.pop()


def build_basis_from_features(infix_features, feature_names, predictors, variable_type_indices):
    basis = np.zeros((predictors.shape[0], len(infix_features)))
    for j, f in enumerate(infix_features):
        if variable_type_indices and f.startswith('RangeOperation'):
            range_operation = RangeOperation(variable_type_indices, feature_names, predictors, string=f)
            basis[:, j] = np.squeeze(range_operation.value)
        elif f in feature_names:
            variable_index = feature_names.index(f)
            basis[:, j] = predictors[:, variable_index]
        else:
            operation_stack = build_operation_stack(f)
            basis[:, j] = get_feature_value(operation_stack, feature_names, predictors, variable_type_indices)
    return basis


def get_basis_from_infix_features(infix_features, feature_names, predictors, scaler=None,
                                  variable_type_indices=None):
    basis = build_basis_from_features(infix_features, feature_names, predictors, variable_type_indices)
    basis = np.nan_to_num(basis)
    if scaler:
        basis = scaler.transform(basis)
    return basis


def optimize(predictors, response, seed, fitness_algorithm, max_gens=100, num_additions=None, preserve_originals=True,
             tournament_probability=.9, max_useless_steps=10, fitness_threshold=.01, correlation_threshold=0.95,
             reinit_range_operators=3, splits=3, time_series_cv=False, feature_names=None, range_operators=0,
             variable_type_indices=None, verbose=False):
    assert predictors.shape[1] == len(feature_names)
    num_additions, feature_names = init(num_additions, feature_names, predictors, seed)
    features = init_features(feature_names, predictors, preserve_originals, range_operators,
                             variable_type_indices)
    best_models = []
    best_features = []
    best_scalers = []
    best_validation_scores = []
    statistics = Statistics()
    best_score = np.Inf
    steps_without_new_model = 0
    response_variance = np.var(response)
    gen = 1
    while gen <= max_gens and steps_without_new_model <= max_useless_steps:
        print('Generation: ' + str(gen))
        score, model, scaler = score_model(features, response, time_series_cv, splits)
        statistics.add(gen, score, len(features))
        if verbose:
            print(get_model_string(features))
        print('Score: ' + str(score))
        if score < best_score:
            best_validation_scores.append(score)
            steps_without_new_model = 0
            best_score = score
            print('New best model score: ' + str(best_score))
            best_models.append(model)
            temp_features = deepcopy(features)
            for f in temp_features:
                f.value = None
            best_features.append(temp_features)
            best_scalers.append(scaler)
        else:
            steps_without_new_model += 1
        print('-------------------------------------------------------')
        if gen < max_gens and steps_without_new_model <= max_useless_steps:
            features = compose_features(num_additions, features, tournament_probability, correlation_threshold,
                                        range_operators, verbose)
            features = update_fitness(features, response, fitness_threshold, fitness_algorithm,
                                      response_variance, num_additions, time_series_cv, splits,
                                      verbose)
            if gen % reinit_range_operators == 0:
                features = swap_range_operators(features, range_operators, variable_type_indices, feature_names,
                                                predictors)
        gen += 1
    return statistics, best_models, best_features, best_scalers, best_validation_scores


def swap_range_operators(features, range_operations, variable_type_indices, feature_names, predictors):
    for f in features:
        if type(f) == RangeOperation and f.original_variable:
            features.remove(f)
    for _ in range(range_operations):
        features.append(RangeOperation(variable_type_indices, feature_names, predictors))
    return features


def name_operation(operation, name):
    operation.__name__ = name
    return operation


class RangeOperation(Feature):

    def __init__(self, variable_type_indices, names, predictors, operation=None, begin_range_name=None,
                 end_range_name=None, original_variable=True, string=None):
        Feature.__init__(self, None, 'RangeOperation', 'RangeOperation', original_variable=original_variable)
        self.predictors = predictors
        self.begin_range = None
        self.end_range = None
        self.operation = None
        self.names = None
        self.lower_bound = None
        self.upper_bound = None
        self.variable_type_indices = variable_type_indices
        self.operations = {
            'sum': name_operation(np.sum, 'sum'),
            'min': name_operation(np.min, 'min'),
            'max': name_operation(np.max, 'max'),
            'mean': name_operation(np.mean, 'mean'),
            'vari': name_operation(np.var, 'vari'),
            'skew': name_operation(skew, 'skew')
        }
        if string:
            parts = string.split('_')
            self.initialize_parameters(variable_type_indices, names, parts[1], parts[2], parts[3])
        else:
            self.initialize_parameters(variable_type_indices, names, operation, begin_range_name, end_range_name)
        self.value = self.create_input_vector()
        self.string = self.format()
        self.infix_string = self.format()

    def __deepcopy__(self, memo):
        new = self.__class__(self.variable_type_indices, self.names, self.predictors)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        new.predictors = self.predictors
        new.value = self.value
        return new

    def initialize_parameters(self, variable_type_indices, names, operation=None, begin_range_name=None,
                              end_range_name=None):
        """
        :param variable_type_indices: A sequence of variable type indices where each entry defines the
        index of a variable type in the design matrix. For example a design matrix with two variable types will have
        indices [j,n] where variable type A spans 0 to j and variable type B spans j + 1 to n.
        :param names:
        :param operation
        :param begin_range_name
        :param end_range_name
        :return:
        """
        self.names = names
        for r in variable_type_indices:
            if r[1] - r[0] < 2:
                raise ValueError('Invalid variable type indices: ' + str(r))
        rng = random.choice(variable_type_indices)
        self.lower_bound = rng[0]
        self.upper_bound = rng[1]
        if operation is not None and begin_range_name is not None and end_range_name is not None:
            if self.operations.get(operation) is None:
                raise ValueError('Invalid operation provided to Range Terminal: ' + operation)
            if begin_range_name not in self.names:
                raise ValueError('Invalid range name provided to Range Termnial: ' + str(begin_range_name))
            if end_range_name not in self.names:
                raise ValueError('Invalid range name provided to Range Terminal: ' + str(end_range_name))
            begin_range = self.names.index(begin_range_name)
            end_range = self.names.index(end_range_name) + 1
            valid = False
            for r in variable_type_indices:
                if r[0] <= begin_range < end_range <= r[1]:
                    valid = True
            if not valid:
                raise ValueError('Invalid range provided to Range Terminal: (' + str(begin_range) + ',' +
                                 str(end_range) + ')')
            self.operation = self.operations[operation]
            self.begin_range = begin_range
            self.end_range = end_range
        else:
            self.operation = random.choice(list(self.operations.values()))
            self.begin_range = np.random.randint(self.lower_bound, self.upper_bound - 1)
            self.end_range = np.random.randint(self.begin_range + 1, self.upper_bound)

    def mutate_parameters(self):
        old = self.format()
        mutation = random.choice(['low', 'high'])
        span = self.end_range - self.begin_range
        if span == 0:
            span = 1
        value = random.gauss(0, math.sqrt(span))
        amount = int(math.ceil(abs(value)))
        if value < 0:
            amount *= -1
        if mutation == 'low':
            location = amount + self.begin_range
            if location < self.lower_bound:
                self.begin_range = self.lower_bound
            elif location > self.end_range - 2:
                self.begin_range = self.end_range - 2
            elif location > self.upper_bound - 2:
                self.begin_range = self.upper_bound - 2
            else:
                self.begin_range = location
        elif mutation == 'high':
            location = amount + self.end_range
            if location > self.upper_bound:
                self.end_range = self.upper_bound
            elif location < self.begin_range + 2:
                self.end_range = self.begin_range + 2
            elif location < self.lower_bound + 2:
                self.end_range = self.lower_bound + 2
            else:
                self.end_range = location
        self.value = self.create_input_vector()
        self.infix_string = self.format()
        self.string = self.format()
        # print('Mutated ' + old + ' to ' + self.format())

    def create_input_vector(self):
        array = self.predictors[:, self.begin_range:self.end_range]
        if array.shape[1] == 0:
            return np.zeros((array.shape[0], 1))
        else:
            return self.operation(array, axis=1)

    def format(self):
        return "RangeOperation_{}_{}_{}".format(self.operation.__name__, self.names[self.begin_range],
                                                self.names[self.end_range - 1])

