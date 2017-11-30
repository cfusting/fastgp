import csv
import operator
from collections import defaultdict
from copy import deepcopy

import numpy

from fastgp.algorithms import afpo
from fastgp.utilities import symbreg


class FitnessDistributionArchive(object):
    def __init__(self, frequency):
        self.fitness = []
        self.generations = []
        self.frequency = frequency
        self.generation_counter = 0

    def update(self, population):
        if self.generation_counter % self.frequency == 0:
            fitnesses = [ind.fitness.values for ind in population]
            self.fitness.append(fitnesses)
            self.generations.append(self.generation_counter)
        self.generation_counter += 1

    def save(self, log_file):
        fitness_distribution_file = "fitness_" + log_file
        with open(fitness_distribution_file, 'wb') as f:
            writer = csv.writer(f)
            for gen, ages in zip(self.generations, self.fitness):
                writer.writerow([gen, ages])


def pick_fitness_size_from_fitness_age_size(ind):
    ind.fitness.values = (ind.error, 0, len(ind))


def pick_fitness_complexity_from_fitness_age_complexity(ind):
    ind.fitness.values = (ind.error, 0, symbreg.calculate_order(ind))


def pick_fitness_size_complexity_from_fitness_age_size_complexity(ind):
    ind.fitness.values = (ind.error, 0, len(ind), symbreg.calculate_order(ind))


def pick_fitness_size_from_fitness_age(ind):
    ind.fitness.values = (ind.error, len(ind))


class MultiArchive(object):
    def __init__(self, archives):
        self.archives = archives

    def update(self, population):
        for archive in self.archives:
            archive.update(population)

    def save(self, log_file):
        for archive in self.archives:
            archive.save(log_file)


class ParetoFrontSavingArchive(object):
    def __init__(self, frequency, criteria_chooser=None, simplifier=None):
        self.fronts = []
        self.frequency = frequency
        self.generation_counter = 0
        self.criteria_chooser = criteria_chooser
        self.simplifier = simplifier

    def update(self, population):
        if self.generation_counter % self.frequency == 0:
            pop_copy = [deepcopy(ind) for ind in population]
            if self.simplifier is not None:
                self.simplifier(pop_copy)
            if self.criteria_chooser is not None:
                map(self.criteria_chooser, pop_copy)

            non_dominated = afpo.find_pareto_front(pop_copy)
            front = [pop_copy[index] for index in non_dominated]
            front.sort(key=operator.attrgetter("fitness.values"))
            self.fronts.append(front)
        self.generation_counter += 1

    def save(self, log_file):
        pareto_front_file = "pareto_" + log_file
        with open(pareto_front_file, 'w') as f:
            writer = csv.writer(f)
            generation = 0
            for front in self.fronts:
                inds = [(ind.fitness.values, str(ind)) for ind in front]
                writer.writerow([generation, len(inds)] + inds)
                generation += self.frequency


class MutationStatsArchive(object):
    def __init__(self, evaluate_function):
        self.stats = defaultdict(list)
        self.neutral_mutations = defaultdict(int)
        self.detrimental_mutations = defaultdict(int)
        self.beneficial_mutations = defaultdict(int)
        self.evaluate_function = evaluate_function
        self.generation = -1

    def update(self, population):
        self.generation += 1

    def submit(self, old_ind, new_ind):
        old_error = self.evaluate_function(old_ind)[0]
        new_error = self.evaluate_function(new_ind)[0]
        delta_error = new_error - old_error
        delta_size = len(new_ind) - len(old_ind)
        if delta_size == 0 and numpy.isclose([delta_error], [0.0])[0]:
            self.neutral_mutations[self.generation] += 1
        if delta_error > 0:
            self.detrimental_mutations[self.generation] += 1
        elif delta_error < 0:
            self.beneficial_mutations[self.generation] += 1
        self.stats[self.generation].append((delta_error, delta_size))

    def save(self, log_file):
        mutation_statistics_file = "mutation_stats_" + log_file
        fieldnames = ['generation', 'neutral_mutations', 'beneficial_mutations', 'detrimental_mutations', 'deltas']
        with open(mutation_statistics_file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            for gen in self.stats.keys():
                writer.writerow([gen, self.neutral_mutations[gen], self.beneficial_mutations[gen],
                                 self.detrimental_mutations[gen]] + self.stats[gen])
