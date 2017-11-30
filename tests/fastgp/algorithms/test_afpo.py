import copy
import operator
import random
import unittest

import cachetools
import numpy
from deap import creator, gp, base, tools

from fastgp.algorithms import afpo, fast_evaluate
from fastgp.logging import reports
from fastgp.utilities import symbreg, benchmark_problems


def get_toolbox(predictors, response):
    creator.create("FitnessAge", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessAge, age=int)

    toolbox = base.Toolbox()
    pset = symbreg.get_numpy_pset(len(predictors[0]))
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=3, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selRandom)

    toolbox.register("mate", gp.cxOnePoint)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=300))

    expression_dict = cachetools.LRUCache(maxsize=10000)
    toolbox.register("error_func", fast_evaluate.normalized_cumulative_absolute_error, response=response,
                     threshold=0.01)
    toolbox.register("evaluate_error", fast_evaluate.fast_numpy_evaluate, context=pset.context, predictors=predictors,
                     error_function=toolbox.error_func, expression_dict=expression_dict)
    toolbox.register("assign_fitness", afpo.assign_age_fitness, tag_depth=2)

    mstats = reports.configure_basic_stats()
    pop = toolbox.population(n=500)
    toolbox.register("run", afpo.pareto_optimization, population=pop, toolbox=toolbox, xover_prob=0.9, mut_prob=0.0, ngen=20,
                     tournament_size=2, num_randoms=1, stats=mstats, calc_pareto_front=False)
    toolbox.register("save", reports.save_log_to_csv)
    return toolbox


def reduce_population(population, tournament_size, target_popsize, nondominated_size):
    while len(population) > target_popsize and len(population) > nondominated_size:
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament = [population[index] for index in tournament_indices]
        # tournament = random.sample(population, tournament_size)
        nondominated_tournament = afpo.find_pareto_front(tournament)
        for i in range(len(tournament)):
            if i not in nondominated_tournament:
                population.remove(tournament[i])


def reduce_population_fix(population, tournament_size, target_popsize, nondominated_size):
    new_population_indices = list(range(len(population)))
    while len(new_population_indices) > target_popsize and len(new_population_indices) > nondominated_size:
        tournament_indices = random.sample(new_population_indices, tournament_size)
        tournament = [population[index] for index in tournament_indices]
        nondominated_tournament = afpo.find_pareto_front(tournament)
        for i in range(len(tournament)):
            if i not in nondominated_tournament:
                new_population_indices.remove(tournament_indices[i])
    population[:] = [population[i] for i in new_population_indices]


class ReducePopulationTestCase(unittest.TestCase):
    def setUp(self):
        random.seed(111111)
        numpy.random.seed(111111)

        p, r = benchmark_problems.get_training_set(benchmark_problems.mod_quartic, num_points=20)
        self.toolbox = get_toolbox(p, r)
        self.pop = self.toolbox.population(n=1000)
        for ind in self.pop:
            ind.age = random.randint(0, 100)
            ind.error = self.toolbox.evaluate_error(ind)[0]
            ind.fitness.values = (ind.error, ind.age)

    def test_reduce_population(self):
        pop_copy = copy.deepcopy(self.pop)
        self.assertListEqual(self.pop, pop_copy)
        random.seed(111)
        afpo.reduce_population_pairwise(self.pop, 2, 500, 0)
        random.seed(111)
        afpo.reduce_population_fix(pop_copy, 2, 500, 0)
        self.assertListEqual(self.pop, pop_copy)

    def test_compare_reduce(self):
        pop_copy = copy.deepcopy(self.pop)
        pop_copy.sort()
        self.pop.sort()
        while len(self.pop) > 500:
            print(len(self.pop))
            self.assertListEqual(self.pop, pop_copy)
            tournament_indices = random.sample(range(len(self.pop)), 2)
            tournament1 = [self.pop[index] for index in tournament_indices]
            tournament2 = [pop_copy[index] for index in tournament_indices]
            self.assertListEqual(tournament1, tournament2)
            self.assertListEqual(self.pop, pop_copy)
            nondominated_tournament1 = afpo.find_pareto_front(tournament1)
            nondominated_tournament2 = afpo.find_pareto_front(tournament2)
            self.assertSetEqual(nondominated_tournament1, nondominated_tournament2)
            for i in range(len(tournament1)):
                if i not in nondominated_tournament1:
                    self.pop.pop(tournament_indices[i])
                    pop_copy.remove(tournament2[i])
                    self.assertListEqual(self.pop, pop_copy)


if __name__ == '__main__':
    unittest.main()
