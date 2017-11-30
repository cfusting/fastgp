import copy
from functools import wraps
import random


def static_limit(key, max_value):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [copy.deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                if key(ind) > max_value:
                    new_inds[i] = copy.deepcopy(random.choice(keep_inds))
            return new_inds
        return wrapper
    return decorator


def stats_collector(archive):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [copy.deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for old_ind, new_ind in zip(keep_inds, new_inds):
                archive.submit(old_ind, new_ind)
            return new_inds
        return wrapper
    return decorator


def internally_biased_node_selector(individual, bias):
    internal_nodes = []
    leaves = []

    for index, node in enumerate(individual):
        if node.arity == 0:
            leaves.append(index)
        else:
            internal_nodes.append(index)

    if internal_nodes and random.random() < bias:
        return random.choice(internal_nodes)
    else:
        return random.choice(leaves)


def get_node_indices_at_depth(individual, level):
    stack = [0]
    nodes_at_depth = []
    for index, node in enumerate(individual):
        current_depth = stack.pop()
        if current_depth == level:
            nodes_at_depth.append(index)
        stack.extend([current_depth + 1] * node.arity)

    return nodes_at_depth


def uniform_depth_node_selector(individual):
    depth = random.randint(0, individual.height)
    nodes_at_depth = get_node_indices_at_depth(individual, depth)
    return random.choice(nodes_at_depth)


def uniform_depth_mutation(individual, expr, pset):
    node_index = uniform_depth_node_selector(individual)
    slice_ = individual.searchSubtree(node_index)
    type_ = individual[node_index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual,


def multi_mutation(ind, mutations, probs):
    for mutation, probability in zip(mutations, probs):
        if random.random() < probability:
            ind, = mutation(ind)
    return ind,


def one_point_xover_biased(ind1, ind2, node_selector):
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    index1 = node_selector(ind1)
    index2 = node_selector(ind2)
    slice1 = ind1.searchSubtree(index1)
    slice2 = ind2.searchSubtree(index2)
    ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


def mutation_biased(ind, expr, node_selector):
    index = node_selector(ind)
    slice1 = ind.searchSubtree(index)
    ind[slice1] = expr()
    return ind,


def static_limit_retries(key, max_value, num_retries):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [copy.deepcopy(ind) for ind in args]

            for _ in range(num_retries):
                new_inds = list(func(*args, **kwargs))
                all_within_limit = True
                for i, ind in enumerate(new_inds):
                    if key(ind) > max_value:
                        all_within_limit = False
                        break
                if all_within_limit:
                    return new_inds

            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                if key(ind) > max_value:
                    new_inds[i] = random.choice(keep_inds)
            return new_inds
        return wrapper
    return decorator
