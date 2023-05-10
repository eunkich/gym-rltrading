import math


def round_val(val, min_val, max_val):
    val = math.tanh(val) + 1
    val *= (max_val - min_val) / 2
    val += min_val
    return round(val)


def round_inv(val, min_val, max_val):
    val -= min_val
    val /= (max_val - min_val) / 2
    val = math.atanh(val - 1)
    return val
