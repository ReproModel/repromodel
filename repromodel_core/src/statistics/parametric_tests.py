import numpy as np
from scipy import stats

def paired_t_test(data1, data2, alpha=0.05):
    statistic, p_value = stats.ttest_rel(data1, data2)
    return {
        'test': 'Paired t-test',
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha
    }

def two_sample_t_test(data1, data2, alpha=0.05, equal_var=True):
    statistic, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
    return {
        'test': 'Two-sample t-test',
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha
    }

def one_way_anova(*args):
    f_statistic, p_value = stats.f_oneway(*args)
    return {
        'test': 'One-way ANOVA',
        'f_statistic': f_statistic,
        'p_value': p_value
    }

def repeated_measures_anova(*args):
    # Placeholder implementation
    # Actual implementation requires more complex calculations
    return {
        'test': 'Repeated Measures ANOVA',
        'f_statistic': None,
        'p_value': None
    }

def linear_regression(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {
        'test': 'Linear Regression',
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }

def f_test(data1, data2):
    f = np.var(data1, ddof=1) / np.var(data2, ddof=1)
    dfn, dfd = len(data1) - 1, len(data2) - 1
    p_value = 1 - stats.f.cdf(f, dfn, dfd)
    return {
        'test': 'F-test for comparing variances',
        'f_statistic': f,
        'p_value': p_value
    }

def z_test(data, population_mean, population_std):
    z_statistic = (np.mean(data) - population_mean) / (population_std / np.sqrt(len(data)))
    p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
    return {
        'test': 'Z-test',
        'z_statistic': z_statistic,
        'p_value': p_value
    }