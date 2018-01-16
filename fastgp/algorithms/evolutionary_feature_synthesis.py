import random
import math

import numpy as np

from sklearn.linear_model import ElasticNetCV

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

    def __init__(self, value, string, infix_string, size=0, fitness=1):
        self.value = value
        self.fitness = fitness
        self.string = string
        self.infix_string = infix_string
        self.size = size

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
    Operator(np.add, 2, '({0} + {1})', '(add({0},{1}))', 'add'),
    Operator(np.subtract, 2, '({0} - {1})', '(sub({0},{1}))', 'sub'),
    Operator(np.multiply, 2, '({0} * {1})', '(mul({0},{1}))', 'mul'),
    Operator(numpy_protected_div_dividend, 2, '{0} / {1}', '(div({0},{1}))', 'div'),
    # Operator(numpy_safe_exp, 1, 'exp({0})'),
    Operator(numpy_protected_log_one, 1, 'log({0})', 'log({0})', 'log'),
    Operator(square, 1, 'sqr({0})', 'sqr({0})', 'sqr'),
    Operator(numpy_protected_sqrt, 1, 'sqt({0})', 'sqt({0})', 'sqt'),
    Operator(cube, 1, 'cbe({0})', 'cbe({0})', 'cbe'),
    Operator(np.cbrt, 1, 'cbt({0})', 'cbt({0})', 'cbt')

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


def init_features(feature_names, predictors):
    features = []
    for i, name in enumerate(feature_names):
        features.append(Feature(predictors[:, i], name))
    return features


def get_basis(features):
    basis = np.zeros((features[0].value.shape[0], len(features)))
    for i, f in enumerate(features):
        basis[:, i] = features[i].value
    basis = np.nan_to_num(basis)
    if np.any(np.isnan(basis)):
        print('Warning: NaN values detected.')
    if not np.all(np.isfinite(basis)):
        print('Warning: Non-finite values detected.')
    return basis


def get_model(basis, response):
    model = ElasticNetCV(normalize=True)
    model.fit(basis, response)
    return model


def tournament_selection(population, probability):
    individuals = random.choices(population, k=2)
    individuals.sort(reverse=True, key=lambda x: x.fitness)
    if random.random() < probability:
        return individuals[0]
    else:
        return individuals[1]


def get_selected_features(num_additions, features, tournament_probability):
    selected_features = []
    for _ in range(num_additions):
        feature_index = tournament_selection(features, tournament_probability)
        selected_features.append(feature_index)
    return selected_features


def update_fitness(features, response, threshold, verbose):
    basis = get_basis(features)
    model = get_model(basis, response)
    remove_features = []
    for i, coef in enumerate(model.coef_):
        fitness = math.fabs(coef)
        features[i].fitness = fitness
        if fitness < threshold:
            remove_features.append(features[i])
    for f in remove_features:
        features.remove(f)
    if verbose and remove_features:
        print('Removed ' + str(len(remove_features)) + ' features from population.')
        print(get_model_string(remove_features))


def compose_features(num_additions, features, tournament_probability, verbose):
    selected_features = get_selected_features(num_additions, features, tournament_probability)
    new_feature_list = []
    for _ in range(num_additions):
        operator = random.choice(operators)
        if operator.parity == 1:
            new_feature = random.choice(selected_features)
            new_feature_string = operator.string.format(new_feature.string)
            new_infix_string = operator.infix.format(new_feature.infix)
            new_feature_value = operator.operation(new_feature.value)
            new_feature_list.append(Feature(new_feature_value, new_feature_string, new_infix_string,
                                            size=new_feature.size + 1))
        elif operator.parity == 2:
            new_features = random.choices(selected_features, k=2)
            new_feature_string = operator.string.format(new_features[0].string, new_features[1].string)
            new_infix_string = operator.infix.format(new_features[0].infix, new_features[1].infix)
            new_feature_value = operator.operation(new_features[0].value, new_features[1].value)
            new_feature_list.append(Feature(new_feature_value, new_feature_string, new_infix_string,
                                            size=new_features[0].size + new_features[1].size + 1))
    filtered_feature_list = list(filter(lambda x: x.size < 5, new_feature_list))
    features.extend(filtered_feature_list)
    if verbose:
        print('Adding ' + str(len(filtered_feature_list)) + ' features to population.')
        print(get_model_string(new_feature_list))


def score_model(features, response, verbose):
    if verbose:
        print('Scoring model with ' + str(len(features)) + ' features.')
    basis = get_basis(features)
    model = get_model(basis, response)
    score = mean_squared_error(model.predict(basis), response)[0]
    return score, model


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
            operator = operators_map[substring]
            stack.append(operator)
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


def get_feature_value(stack, feature_names, predictors):
    variables_stack = []
    while len(stack) > 1:
        current = stack.pop()
        if current in feature_names:
            variables_stack.append(current)
        elif current in operators_map:
            operator = operators_map[current]
            variables = []
            for _ in operator.parity:
                variables.append(variables_stack.pop())
            # Double check this happens in the right order
            result = operator.operation(variables_stack)
            variables_stack.append(result)


def build_basis_from_features(features, feature_names, predictors):
    basis = np.zeros((features[0].value.shape[0], len(features)))
    for j, f in enumerate(features):
        operation_stack = build_operation_stack(f.infix_string)
        basis[:, j] = get_feature_value(operation_stack, feature_names, predictors)
        return basis


def build_model_from_data(features, feature_names, predictors, response):
    basis = build_basis_from_features(features, feature_names, predictors)
    model = get_model(basis, response)


def optimize(predictors, response, max_gens, seed, num_additions=None, tournament_probability=.9,
             max_useless_steps=10, fitness_threshold=.01, feature_names=None, verbose=False):
    assert predictors.shape[1] == len(feature_names)
    num_additions, feature_names = init(num_additions, feature_names, predictors, seed)
    models = []
    statistics = Statistics()
    best_score = np.Inf
    steps_without_new_model = 0
    features = init_features(feature_names, predictors)
    gen = 1
    while gen < max_gens and steps_without_new_model < max_useless_steps:
        if verbose:
            print('Generation: ' + str(gen))
        score, model = score_model(features, response, verbose)
        statistics.add(gen, score, len(features))
        print(get_model_string(features))
        print('Score: ' + str(score))
        if score < best_score:
            steps_without_new_model = 0
            best_score = score
            print('New best model score: ' + str(best_score))
            models.append(model)
        else:
            steps_without_new_model += 1
        compose_features(num_additions, features, tournament_probability, verbose)
        update_fitness(features, response, fitness_threshold, verbose)
        gen += 1
        if verbose:
            print('-------------------------------------------------------')
    return statistics, models, features
