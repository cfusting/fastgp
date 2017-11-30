import random
from functools import partial
import math
import re

import cachetools
import numpy as np
from scipy.stats import skew, moment
from copy import deepcopy

from deap import gp


class SimpleParametrizedPrimitiveSet(gp.PrimitiveSet):
    def __init__(self, name, arity, variable_type_indices, variable_names, prefix="ARG"):
        gp.PrimitiveSet.__init__(self, name, arity, prefix)
        self.variable_type_indices = variable_type_indices
        self.variable_names = variable_names

    def add_parametrized_terminal(self, parametrized_terminal_class):
        self._add(parametrized_terminal_class)
        self.context[parametrized_terminal_class.__name__] = parametrized_terminal_class.call


class SimpleParametrizedPrimitiveTree(gp.PrimitiveTree):
    def __init__(self, content):
        gp.PrimitiveTree.__init__(self, content)

    def __deepcopy__(self, memo):
        new = self.__class__(self)
        for i, node in enumerate(self):
            if isinstance(node, SimpleParametrizedTerminal):
                new[i] = deepcopy(node)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    @classmethod
    def from_string(cls, string, pset):
        """Try to convert a string expression into a PrimitiveTree given a
        PrimitiveSet *pset*. The primitive set needs to contain every primitive
        present in the expression.

        :param string: String representation of a Python expression.
        :param pset: Primitive set from which primitives are selected.
        :returns: PrimitiveTree populated with the deserialized primitives.
        """
        tokens = re.split("[ \t\n\r\f\v(),]", string)
        expr = []

        def get_parts(token_string):
            parts = tokens[i].split('_')
            return parts[1], parts[2], parts[3]
        i = 0
        while i < len(tokens):
            if tokens[i] == '':
                i += 1
                continue
            if tokens[i] in pset.mapping:
                primitive = pset.mapping[tokens[i]]
                expr.append(primitive)
            elif RangeOperationTerminal.NAME in tokens[i]:
                operation, begin_range_name, end_range_name = get_parts(tokens[i])
                range_operation_terminal = RangeOperationTerminal()
                range_operation_terminal.initialize_parameters(pset.variable_type_indices, pset.variable_names,
                                                               operation, begin_range_name, end_range_name)
                expr.append(range_operation_terminal)
            elif MomentFindingTerminal.NAME in tokens[i]:
                operation, begin_range_name, end_range_name = get_parts(tokens[i])
                moment_operation_terminal = MomentFindingTerminal()
                moment_operation_terminal.initialize_parameters(pset.variable_type_indices, pset.variable_names,
                                                                operation, begin_range_name, end_range_name)
                expr.append(moment_operation_terminal)
            else:
                try:
                    token = eval(tokens[i])
                except NameError:
                    raise TypeError("Unable to evaluate terminal: {}.".format(tokens[i]))
                expr.append(gp.Terminal(token, False, gp.__type__))
            i += 1
        return cls(expr)


class SimpleParametrizedTerminal(gp.Terminal):
    ret = object

    def __init__(self, name="SimpleParametrizedTerminal", ret_type=object):
        gp.Terminal.__init__(self, name, True, ret_type)

    def __deepcopy__(self, memo):
        new = self.__class__()
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    def initialize_parameters(self, variable_type_indices, names):
        raise NotImplementedError

    def create_input_vector(self, predictors):
        raise NotImplementedError

    def call(*parameters):
        pass  # implement this method to make the class work with standard gp.compile


def name_operation(operation, name):
    operation.__name__ = name
    return operation


class RangeOperationTerminal(SimpleParametrizedTerminal):
    NAME = 'RangeOperation'

    def __init__(self):
        SimpleParametrizedTerminal.__init__(self, RangeOperationTerminal.__name__)
        self.begin_range = None
        self.end_range = None
        self.operation = None
        self.names = None
        self.lower_bound = None
        self.upper_bound = None
        self.operations = {
            'sum': name_operation(np.sum, 'sum'),
            'min': name_operation(np.min, 'min'),
            'max': name_operation(np.max, 'max')
        }

    def initialize_parameters(self, variable_type_indices, names, operation=None, begin_range_name=None,
                              end_range_name=None, *args):
        """
        :param variable_type_indices: A sequence of variable type indices where each entry defines the
        index of a variable type in the design matrix. For example a design matrix with two variable types will have
        indices [j,n] where variable type A spans 0 to j and variable type B spans j + 1 to n.
        :param names:
        :param args:
        :param operation
        :param begin_range_name
        :param end_range_name
        :return:
        """
        self.names = names
        for r in variable_type_indices:
            if r[1] - r[0] < 2:
                raise ValueError('Invalid range provided to Range Terminal: ' + str(r))
        rng = random.choice(variable_type_indices)
        self.lower_bound = rng[0]
        self.upper_bound = rng[1]
        if operation is not None and begin_range_name is not None and end_range_name is not None:
            if self.operations.get(operation) is None:
                raise ValueError('Invalid operation provided to Range Terminal: ' + operation)
            if begin_range_name not in self.names:
                raise ValueError('Invalid range name provided to Range Termnial: ' + str(begin_range_name))
            if end_range_name not in names:
                raise ValueError('Invalid range name provided to Range Termnial: ' + str(end_range_name))
            begin_range = self.names.index(begin_range_name)
            end_range = self.names.index(end_range_name)
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

    def mutate_parameters(self, stdev_calc):
        mutation = random.choice(['low', 'high'])
        span = self.end_range - self.begin_range
        if span == 0:
            span = 1
        value = random.gauss(0, stdev_calc(span))
        amount = int(math.ceil(abs(value)))
        if value < 0:
            amount *= -1
        if mutation == 'low':
            self.begin_range += amount
            if self.begin_range < self.lower_bound:
                self.begin_range = self.lower_bound
            elif self.begin_range >= self.end_range:
                self.begin_range = self.end_range - 2
        else:
            self.end_range += amount
            if self.end_range >= self.upper_bound:
                self.end_range = self.upper_bound
            elif self.end_range <= self.begin_range:
                self.end_range = self.begin_range + 2

    def create_input_vector(self, predictors):
        return self.operation(predictors[:, self.begin_range:self.end_range], axis=1)

    def format(self):
        return "RangeOperation_{}_{}_{}".format(self.operation.__name__, self.names[self.begin_range],
                                                self.names[self.end_range - 1])


class MomentFindingTerminal(RangeOperationTerminal):
    NAME = 'MomentOperation'

    def __init__(self):
        super(MomentFindingTerminal, self).__init__()
        self.operations = {
            'mean': name_operation(np.mean, 'mean'),
            'vari': name_operation(np.var, 'vari'),
            'skew': name_operation(skew, 'skew')
        }

    def initialize_parameters(self, variable_type_indices, names, operation=None, begin_range_name=None,
                              end_range_name=None, *args):
        if operation is None:
            super(MomentFindingTerminal, self).initialize_parameters(variable_type_indices, names)
            self.operation = random.choice(self.operations.values())
        else:
            super(MomentFindingTerminal, self).initialize_parameters(variable_type_indices, names, operation,
                                                                     begin_range_name, end_range_name, *args)

    def format(self):
        return "MomentOperation_{}_{}_{}".format(self.operation.__name__, self.names[self.begin_range],
                                                 self.names[self.end_range])


def named_moment(number):
    def f(vector, axis=0):
        return moment(vector, moment=number, axis=axis)
    f.__name__ = "moment_" + str(number)
    return f


def generate_parametrized_expression(generate_expression, variable_type_indices, names):
    expr = generate_expression()
    for node in expr:
        if isinstance(node, SimpleParametrizedTerminal):
            node.initialize_parameters(variable_type_indices, names)
    return expr


def evolve_parametrized_expression(stdev_calc):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = list(func(*args, **kargs))
            for ind in offspring:
                for node in ind:
                    if isinstance(node, SimpleParametrizedTerminal):
                        node.mutate_parameters(stdev_calc)
            return offspring
        return wrapper
    return decorator


def get_parametrized_nodes(ind):
    return list(filter(lambda node: isinstance(node, SimpleParametrizedTerminal), ind))


def mutate_parametrized_nodes(ind, stdev_calc):
    param_nodes = get_parametrized_nodes(ind)
    map(lambda node: node.mutate_parameters(stdev_calc), param_nodes)
    return ind,


def mutate_single_parametrized_node(ind, stdev_calc):
    param_nodes = get_parametrized_nodes(ind)
    if len(param_nodes) != 0:
        random.choice(param_nodes).mutate_parameters(stdev_calc)
    return ind,


def search_entire_space(node, evaluate_function):
    fitness = []
    parameters = []
    begin = node.lower_bound
    while begin <= node.upper_bound:
        end = begin + 1
        while end <= node.upper_bound:
            node.begin_range = begin
            node.end_range = end
            fitness.append(evaluate_function())
            parameters.append((begin, end))
            end += 1
        begin += 1
    return parameters, fitness


def optimize_node(node, evaluate_function, optimization_objective_function):
    parameters, fitness = search_entire_space(node, evaluate_function)
    best_value = optimization_objective_function(fitness)
    optimal_index = fitness.index(best_value)
    begin, end = parameters[optimal_index]
    node.begin_range = begin
    node.end_range = end
    return parameters, fitness


def mutate_single_parametrized_node_optimal(ind, evaluate_function, optimization_objective_function):
    param_nodes = get_parametrized_nodes(ind)
    if len(param_nodes) != 0:
        node = random.choice(param_nodes)
        optimize_node(node, partial(evaluate_function, ind=ind), optimization_objective_function)
    return ind,


def simple_parametrized_evaluate(ind, context, predictors, error_function=None, expression_dict=None):
    semantics_stack = []
    expressions_stack = []

    if expression_dict is None:
        expression_dict = cachetools.LRUCache(maxsize=100)

    for node in reversed(ind):
        expression = node.format(*[expressions_stack.pop() for _ in range(node.arity)])
        subtree_semantics = [semantics_stack.pop() for _ in range(node.arity)]

        if expression in expression_dict:
            vector = expression_dict[expression]
        else:
            vector = get_node_semantics(node, subtree_semantics, predictors, context)
            expression_dict[expression] = vector

        expressions_stack.append(expression)
        semantics_stack.append(vector)

    if error_function is None:
        return semantics_stack.pop()
    else:
        return error_function(semantics_stack.pop())


def get_terminal_semantics(node, context, predictors):
    if isinstance(node, gp.Ephemeral) or isinstance(node.value, float) or isinstance(node.value, int):
        return np.ones(len(predictors)) * node.value

    if node.value in context:
        return np.ones(len(predictors)) * context[node.value]

    arg_index = re.findall('\d+', node.name)
    return predictors[:, int(arg_index[0])]


def get_node_semantics(node, subtree_semantics, predictors, context):
    if isinstance(node, SimpleParametrizedTerminal):
        vector = node.create_input_vector(predictors)
    elif isinstance(node, gp.Terminal):
        vector = get_terminal_semantics(node, context, predictors)
    else:
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            vector = context[node.name](*list(map(lambda x: x.astype(float), subtree_semantics)))
    return vector


def graph(expr):
    nodes = range(len(expr))
    edges = list()
    labels = dict()

    stack = []
    for i, node in enumerate(expr):
        if stack:
            edges.append((stack[-1][0], i))
            stack[-1][1] -= 1
        if isinstance(node, gp.Primitive):
            labels[i] = node.name
        elif isinstance(node, SimpleParametrizedTerminal):
            labels[i] = node.format()
        else:
            labels[i] = node.value
        stack.append([i, node.arity])
        while stack and stack[-1][1] == 0:
            stack.pop()
    return nodes, edges, labels
