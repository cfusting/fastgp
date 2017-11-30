from __future__ import division
import csv
from deap import tools
import numpy
import operator
import fastgp.parametrized.simple_parametrized_terminals as sp


def get_fitness(ind):
    return ind.fitness.values[0]


def get_mean(values):
    return numpy.mean(list(filter(numpy.isfinite, values)))


def get_std(values):
    return numpy.std(list(filter(numpy.isfinite, values)))


def get_min(values):
    return numpy.min(list(filter(numpy.isfinite, values)))


def get_max(values):
    return numpy.max(list(filter(numpy.isfinite, values)))


def get_size_min(values):
    return min(values)[1]


def get_size_max(values):
    return max(values)[1]


def get_fitness_size(ind):
    return ind.fitness.values[0], len(ind)


def configure_inf_protected_stats():
    stats_fit = tools.Statistics(get_fitness)
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(operator.attrgetter("height"))
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, height=stats_height)
    mstats.register("avg", get_mean)
    mstats.register("std", get_std)
    mstats.register("min", get_min)
    mstats.register("max", get_max)

    stats_best_ind = tools.Statistics(get_fitness_size)
    stats_best_ind.register("size_min", get_size_min)
    stats_best_ind.register("size_max", get_size_max)
    mstats["best_tree"] = stats_best_ind
    return mstats


def is_parametrized_terminal(node):
    return isinstance(node, sp.SimpleParametrizedTerminal)


def get_param_ratio(ind):
    parametrized = len(list(filter(is_parametrized_terminal, ind)))
    total = len(ind)
    return parametrized / total


def configure_parametrized_inf_protected_stats():
    stats_fit = tools.Statistics(get_fitness)
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(operator.attrgetter("height"))

    stats_parametrized = tools.Statistics(get_param_ratio)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, height=stats_height,
                                   parametrized=stats_parametrized)
    mstats.register("avg", get_mean)
    mstats.register("std", get_std)
    mstats.register("min", get_min)
    mstats.register("max", get_max)
    stats_best_ind = tools.Statistics(get_fitness_size)
    stats_best_ind.register("size_min", get_size_min)
    stats_best_ind.register("size_max", get_size_max)
    mstats["best_tree"] = stats_best_ind
    return mstats


def get_age(ind):
    return ind.age


def add_age_to_stats(mstats):
    stats_age = tools.Statistics(get_age)
    stats_age.register("avg", numpy.mean)
    stats_age.register("std", numpy.std)
    stats_age.register("max", numpy.max)
    mstats["age"] = stats_age
    return mstats


def save_log_to_csv(pop, log, file_name):
    columns = [log.select("cpu_time")]
    columns_names = ["cpu_time"]
    for chapter_name, chapter in log.chapters.items():
        for column in chapter[0].keys():
            columns_names.append(str(column) + "_" + str(chapter_name))
            columns.append(chapter.select(column))

    rows = zip(*columns)
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(columns_names)
        for row in rows:
            writer.writerow(row)


def save_hof(hof, test_toolbox=None):
    def decorator(func):
        def wrapper(pop, log, file_name):
            func(pop, log, file_name)
            hof_file_name = "trees_" + file_name
            with open(hof_file_name, 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(["gen", "fitness", "tree"])
                for gen, ind in enumerate(hof.historical_trees):
                    if test_toolbox is not None:
                        test_error = test_toolbox.test_evaluate(ind)[0]
                        writer.writerow([gen, ind.fitness, str(ind), test_error])
                    else:
                        writer.writerow([gen, ind.fitness, str(ind)])
        return wrapper
    return decorator


def save_archive(archive):
    def decorator(func):
        def wrapper(pop, log, file_name):
            func(pop, log, file_name)
            archive.save(file_name)
        return wrapper
    return decorator
