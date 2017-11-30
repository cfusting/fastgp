import random


def multi_mutation(ind, mutations, probs):
    for mutation, probability in zip(mutations, probs):
        if random.random() < probability:
            ind = mutation(ind),
    return ind,


def multi_mutation_exclusive(ind, mutations, probs):
    if len(mutations) != len(probs):
        raise ValueError("Must have the same number of mutations as probabilities.")
    if sum(probs) > 1:
        raise ValueError("Probabilities must sum to 1.")
    prob_range = [0] + probs
    value = random.random()
    i = 1
    while i < len(prob_range):
        prob_range[i] += prob_range[i - 1]
        if prob_range[i - 1] <= value < prob_range[i]:
            mutations[i - 1](ind)
            return ind,
        i += 1
    return ind,
