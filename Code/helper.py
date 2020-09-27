from scipy.integrate import quad
import scipy.stats
import numpy as np

def normal_distribution_function(x, mean=0, std=0.25):
    value = scipy.stats.norm.pdf(x, 0, 0.25)
    return value

def get_intervals(res):
    # Normal Distribution
    total_intervals = 2 * res
    x_min = -1
    x_max = 1

    interval_spacing = 2 / float(2 * res)
    x = np.linspace(x_min, x_max, 100)
    
    intervals = []
    count = 0
    start = -1
    while count < (2 * res):
        result, err = quad(normal_distribution_function, -1, start)
        intervals.append(result)
        count = count + 1
        start = start + interval_spacing
    
    final = {}
    count = 1
    intervals.append(1)

    for i in range(0, len(intervals)):
        intervals[i] = -1 + 2 * intervals[i]

    for i in range(0, len(intervals)-1):
        if i == len(intervals) - 1:
            final[count] = [intervals[i], 1]
            continue

        final[count] = [intervals[i], intervals[i + 1]-0.000000000000000001]
        count += 1

    return final