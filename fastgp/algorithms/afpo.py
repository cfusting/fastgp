import logging
import random
import time

from deap import tools

from fastgp.utilities import symbreg


def breed(parents, toolbox, xover_prob, mut_prob):
    offspring = [toolbox.clone(ind) for ind in parents]

    for i in range(1, len(offspring), 2):
        if random.random() < xover_prob:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            max_age = max(offspring[i - 1].age, offspring[i].age)
            offspring[i].age = offspring[i - 1].age = max_age
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mut_prob:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def find_pareto_front(population):
    """Finds a subset of nondominated individuals in a given list

    :param population: a list of individuals
    :return: a set of indices corresponding to nondominated individuals
    """

    pareto_front = set(range(len(population)))

    for i in range(len(population)):
        if i not in pareto_front:
            continue

        ind1 = population[i]
        for j in range(i + 1, len(population)):
            ind2 = population[j]

            # if individuals are equal on all objectives, mark one of them (the first encountered one) as dominated
            # to prevent excessive growth of the Pareto front
            if ind2.fitness.dominates(ind1.fitness) or ind1.fitness == ind2.fitness:
                pareto_front.discard(i)

            if ind1.fitness.dominates(ind2.fitness):
                pareto_front.discard(j)

    return pareto_front


def reduce_population(population, tournament_size, target_popsize, nondominated_size):
    num_iterations = 0
    new_population_indices = list(range(len(population)))
    while len(new_population_indices) > target_popsize and len(new_population_indices) > nondominated_size:
        if num_iterations > 10e6:
            print("Pareto front size may be exceeding the size of population. Stopping the execution. Try making"
                  "the population size larger or the number of generations smaller.")
            # random.sample(new_population_indices, len(new_population_indices) - target_popsize)
            exit()
        num_iterations += 1
        tournament_indices = random.sample(new_population_indices, tournament_size)
        tournament = [population[index] for index in tournament_indices]
        nondominated_tournament = find_pareto_front(tournament)
        for i in range(len(tournament)):
            if i not in nondominated_tournament:
                new_population_indices.remove(tournament_indices[i])
    population[:] = [population[i] for i in new_population_indices]


def pareto_optimization(population, toolbox, xover_prob, mut_prob, ngen, tournament_size, num_randoms=1, archive=None,
                        stats=None, calc_pareto_front=True, verbose=False, reevaluate_population=False, history=None):
    start = time.time()
    if history is not None:
        history.update(population)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'cpu_time'] + (stats.fields if stats else [])

    target_popsize = len(population)

    # calculating errors may be expensive, so we will cache the error value as an individual's attribute
    for ind in population:
        ind.error = toolbox.evaluate_error(ind)[0]
    toolbox.assign_fitness(population)
    for ind in population:
        history.genealogy_history[ind.history_index].error = ind.error

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), cpu_time=time.time() - start, **record)
    if archive is not None:
        archive.update(population)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        # do we want to enforce re-evaluating the whole population instead of using cached erro r values
        if reevaluate_population:
            for ind in population:
                ind.error = toolbox.evaluate_error(ind)[0]
        parents = toolbox.select(population, len(population) - num_randoms)
        offspring = breed(parents, toolbox, xover_prob, mut_prob)
        offspring += toolbox.generate_randoms()

        # evaluate newly generated individuals which do not have cached values (or have inherited them from parents)
        for ind in offspring:
            ind.error = toolbox.evaluate_error(ind)[0]

        # extend the population by adding offspring - the size of population is now 2*target_popsize
        population.extend(offspring)
        toolbox.assign_fitness(population)

        for ind in population:
            history.genealogy_history[ind.history_index].error = ind.error

        # we may take 2 strategies of evaluating pareto-front:
        #   - pessimistic: Pareto front may be larger than target_popsize and we want to detect it early because
        #                  if that's the case we won't be able to reduce the size of population to target_popsize
        #   - optimistic: in practice, the above case happen extremely rarely but calculating global front is expensive
        #                 so let's assume that Pareto front is small enough try to reduce the population
        if calc_pareto_front:
            pareto_front_size = len(find_pareto_front(population))
            logging.debug("Generation: %5d - Pareto Front Size: %5d", gen, pareto_front_size)
            if pareto_front_size > target_popsize:
                logging.info("Pareto front size exceeds the size of population. Try Making the population size larger"
                             "or reducing the number of generations.")
                break
        else:
            pareto_front_size = 0

        # perform Pareto tournament selection until the size of the population is reduced to target_popsize
        reduce_population(population, tournament_size, target_popsize, pareto_front_size)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), cpu_time=time.time() - start, **record)
        if archive is not None:
            archive.update(population)
        if verbose:
            print(logbook.stream)

        for ind in population:
            ind.age += 1

    return population, logbook, history


def evaluate_age_fitness(ind, error_func):
    ind.error = error_func(ind)[0]
    return ind.error, ind.age


def evaluate_age_fitness_size(ind, error_func):
    ind.size = len(ind)
    return evaluate_age_fitness(ind, error_func) + (ind.size,)


def evaluate_fitness_size(ind, error_func):
    ind.error = error_func(ind)[0]
    ind.size = len(ind)
    return ind.error, ind.size


def evaluate_fitness_size_complexity(ind, error_func):
    ind.error = error_func(ind)[0]
    ind.size = len(ind)
    ind.complexity = symbreg.calculate_order(ind)
    return ind.error, ind.size, ind.complexity


def assign_random_fitness(population, random_range):
    for ind in population:
        ind.fitness.values = (ind.error, random.randrange(random_range))


def assign_pure_fitness(population):
    for ind in population:
        ind.fitness.values = (ind.error,)


def assign_age_fitness(population):
    for ind in population:
        ind.fitness.values = (ind.error, ind.age)


def assign_age_fitness_size(population):
    for ind in population:
        ind.fitness.values = (ind.error, ind.age, len(ind))


def assign_age_fitness_complexity(population):
    for ind in population:
        ind.fitness.values = (ind.error, ind.age, symbreg.calculate_order(ind))


def assign_age_fitness_size_complexity(population):
    for ind in population:
        ind.fitness.values = (ind.error, ind.age, len(ind), symbreg.calculate_order(ind))


def assign_size_fitness(population):
    for ind in population:
        ind.fitness.values = (ind.error, len(ind))

