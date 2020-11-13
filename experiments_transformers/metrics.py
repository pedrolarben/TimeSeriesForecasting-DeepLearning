import numpy as np

EPSILON = 1e-10


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error

    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)


def _naive_forecasting(actual: np.ndarray, seasonality: int = 1):
    """ Naive forecasting method which just repeats previous samples """
    return actual[:-seasonality]


def _relative_error(
    actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None
):
    """ Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark
        return _error(actual[seasonality:], predicted[seasonality:]) / (
            _error(actual[seasonality:], _naive_forecasting(actual, seasonality))
            + EPSILON
        )

    return _error(actual, predicted) / (_error(actual, benchmark) + EPSILON)


def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(actual, predicted)))


def wape(actual: np.ndarray, predicted: np.ndarray):
    """ Weighted Absolute Percentage Error """
    return mae(actual, predicted) / (np.mean(actual) + EPSILON)


def mape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Absolute Percentage Error

    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0

    Note: result is NOT multiplied by 100
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))


def smape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Mean Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    return 100 * np.mean(
        2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)))
    )


def mase(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """
    Mean Absolute Scaled Error

    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    return mae(actual, predicted) / (
        mae(actual[seasonality:], _naive_forecasting(actual, seasonality))
    )


def std_ae(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Absolute Error """
    __mae = mae(actual, predicted)
    return np.sqrt(
        np.sum(np.square(_error(actual, predicted) - __mae)) / (len(actual) - 1)
    )


def std_ape(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Absolute Percentage Error """
    __mape = mape(actual, predicted)
    return np.sqrt(
        np.sum(np.square(_percentage_error(actual, predicted) - __mape))
        / (len(actual) - 1)
    )


def mre(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Relative Error """
    # return np.mean( np.abs(_error(actual, predicted)) / (actual + EPSILON))
    return np.mean(_relative_error(actual, predicted, benchmark))


def rae(actual: np.ndarray, predicted: np.ndarray):
    """ Relative Absolute Error (aka Approximation Error) """
    return np.sum(np.abs(actual - predicted)) / (
        np.sum(np.abs(actual - np.mean(actual))) + EPSILON
    )


def mrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Relative Absolute Error """
    return np.mean(np.abs(_relative_error(actual, predicted, benchmark)))


METRICS = {
    # "mse": mse,
    # "rmse": rmse,
    # "mae": mae,
    "wape": wape,
    # "mape": mape,
    # "smape": smape,
    "mase": mase,
    # "std_ae": std_ae,
    # "std_ape": std_ape,
    # "rae": rae,
    #'mre': mre,
    #'mrae': mrae
}

import matplotlib.pyplot as plt


def evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
    metrics=("mae", "mse", "mase", "smape", "wape"),
):
    results = {}
    for name in metrics:
        try:
            res = []

            for y, o in zip(actual, predicted):
                all_zeros = name in ["mape", "wape", "std_ape", "rae"] and np.all(
                    y == 0.0
                )
                all_identical = name in ["mase", "wape", "mape"] and np.all(
                    np.abs(y - np.mean(y)) < 0.001
                )
                if not all_zeros and not all_identical:
                    res.append(METRICS[name](y, o))

                if (
                    False
                    and not all_zeros
                    and not all_identical
                    and METRICS[name](y, o) > 100000
                ):
                    plt.plot(y, label="real")
                    plt.plot(o, label="predicted")
                    plt.title("{}: {}".format(name, METRICS[name](y, o)))
                    plt.legend()
                    plt.show()

            results[name] = np.mean(res)

        except Exception as err:
            results[name] = np.nan
            print("Unable to compute metric {0}: {1}".format(name, err))

    return results


def evaluate_all(actual: np.ndarray, predicted: np.ndarray):
    return evaluate(actual, predicted, metrics=set(METRICS.keys()))
