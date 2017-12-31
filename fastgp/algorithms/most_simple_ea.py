import time
import math
import random

from deap import tools


def generate_next_population(individuals, toolbox):
    """
    Perform truncated selection with elitism.
    See Algorithm 1 from the arXiv preprint arXiv:1712.06567.
    The only difference being that mutation is not defined.
    :param individuals:
    :param toolbox:
    :return:
    """
    individuals = [toolbox.clone(ind) for ind in individuals]
    individuals.sort(key=lambda x: x.error)

    offspring = []
    pop_size = len(individuals)
    num_top = math.floor(pop_size / 2)
    parents = individuals[0:num_top + 1]
    for _ in range(pop_size - 1):
        off = random.choice(parents)
        off = toolbox.mutate(off)[0]
        offspring.append(off)
    offspring.append(individuals[0])
    return offspring


def render_fitness(population, toolbox, history):
    for ind in population:
        ind.error = toolbox.evaluate_error(ind)[0]
        ind.fitness.values = ind.error,
        if history is not None:
            history.genealogy_history[ind.history_index].error = ind.error


def record_information(population, stats, start, archive, logbook, verbose):
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), cpu_time=time.time() - start, **record)
    if archive is not None:
        archive.update(population)
    if verbose:
        print(logbook.stream)


def optimize(population, toolbox, ngen, archive=None, stats=None, verbose=False, history=None):
    """
    Optimize a population of individuals.
    :param population:
    :param toolbox:
    :param mut_prob:
    :param ngen:
    :param archive:
    :param stats:
    :param verbose:
    :param history:
    :return:
    """
    start = time.time()
    if history is not None:
        history.update(population)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'cpu_time'] + (stats.fields if stats else [])

    render_fitness(population, toolbox, history)
    record_information(population, stats, start, archive, logbook, verbose)
    for gen in range(1, ngen + 1):
        offspring = generate_next_population(population, toolbox)
        render_fitness(offspring, toolbox, history)
        population = offspring
        record_information(population, stats, start, archive, logbook, verbose)
    return population, logbook, history

