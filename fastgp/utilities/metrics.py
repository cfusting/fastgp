import numpy
from scipy.stats import pearsonr, spearmanr

from fastgp.utilities.symbreg import numpy_protected_div_dividend


def mean_absolute_error(vector, response):
    errors = numpy.abs(vector - response)
    mean_error = numpy.mean(errors)
    if not numpy.isfinite(mean_error):
        return numpy.inf,
    return mean_error.item(),


def euclidean_error(vector, response):
    with numpy.errstate(over='ignore', divide='ignore', invalid='ignore'):
        squared_errors = numpy.square(vector - response)
    sum_squared_errors = numpy.sum(squared_errors)
    if not numpy.isfinite(sum_squared_errors):
        return numpy.inf,
    distance = numpy.sqrt(sum_squared_errors)
    return distance.item(),


def root_mean_square_error(vector, response):
    with numpy.errstate(over='ignore', divide='ignore', invalid='ignore'):
        squared_errors = numpy.square(vector - response)
    mse = numpy.mean(squared_errors)
    if not numpy.isfinite(mse):
        return numpy.inf,
    rmse = numpy.sqrt(mse)
    return rmse.item(),


def mean_squared_error(vector, response):
    squared_errors = numpy.square(vector - response)
    mse = float(numpy.mean(squared_errors))
    if not numpy.isfinite(mse):
        return numpy.inf,
    return mse,


def pearson_correlation(vector, response):
    return pearsonr(vector, response)


def spearman_correlation(vector, response):
    return spearmanr(vector, response)


def normalized_cumulative_absolute_error(vector, response, threshold=0.0):
    errors = numpy.abs(vector - response)
    raw_sum = numpy.sum(errors)
    if not numpy.isfinite(raw_sum):
        return 0.0,

    errors[errors < threshold] = 0
    cumulative_error = numpy.sum(errors).item()
    return 1 / (1 + cumulative_error),


def mean_absolute_percentage_error(vector, response):
    with numpy.errstate(over='ignore', divide='ignore', invalid='ignore'):
        errors = numpy_protected_div_dividend((vector - response), response)
        errors = numpy_protected_div_dividend(errors, float(len(response)))
    mean_error = numpy.sum(numpy.abs(errors))
    if numpy.isnan(mean_error) or not numpy.isfinite(mean_error):
            return numpy.inf,
    return mean_error,


def percentage_error(vector, response, threshold=0.0):
    errors = numpy.abs(vector - response)
    raw_sum = numpy.sum(errors)
    if not numpy.isfinite(raw_sum):
        return 0.0,

    errors[errors < threshold] = 0
    cumulative_error = numpy.sum(errors).item()
    cumulative_response = numpy.sum(response).item()
    return numpy_protected_div_dividend(cumulative_error, cumulative_response),


def cumulative_absolute_error(vector, response):
    errors = numpy.abs(vector - response)
    cumulative_error = numpy.sum(errors)
    if not numpy.isfinite(cumulative_error):
        return numpy.inf,
    return cumulative_error.item(),


def normalized_mean_squared_error(vector, response):
    squared_errors = numpy.square(vector - response)
    mse = numpy.mean(squared_errors)
    if not numpy.isfinite(mse):
        return numpy.inf,
    normalized_mse = mse / numpy.var(response)
    return normalized_mse.item(),

