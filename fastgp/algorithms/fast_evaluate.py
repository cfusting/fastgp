import cachetools
import numpy


def fast_numpy_evaluate(ind, context, predictors, get_node_semantics, error_function=None, expression_dict=None):
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


def fast_numpy_evaluate_population(pop, context, predictors, error_func, expression_dict=None, arg_prefix="ARG"):
    if expression_dict is None:
        expression_dict = cachetools.LRUCache(maxsize=2000)

    results = numpy.empty(shape=(len(pop), len(predictors)))
    for row, ind in enumerate(pop):
        results[row] = fast_numpy_evaluate(ind, context, predictors, expression_dict, arg_prefix)

    errors = error_func(results)
    for ind, error in zip(pop, errors):
        ind.fitness.values = error,
