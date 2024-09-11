from scipy import stats

def wilcoxon_signed_rank_test(data1, data2, alpha=0.05):
    statistic, p_value = stats.wilcoxon(data1, data2)
    return {
        'test': 'Wilcoxon Signed-Rank Test',
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha
    }

def mann_whitney_u_test(data1, data2, alpha=0.05):
    statistic, p_value = stats.mannwhitneyu(data1, data2)
    return {
        'test': 'Mann-Whitney U Test',
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha
    }

def kruskal_wallis_h_test(*args):
    statistic, p_value = stats.kruskal(*args)
    return {
        'test': 'Kruskal-Wallis H Test',
        'statistic': statistic,
        'p_value': p_value
    }

def friedman_test(*args):
    statistic, p_value = stats.friedmanchisquare(*args)
    return {
        'test': 'Friedman Test',
        'statistic': statistic,
        'p_value': p_value
    }

def spearman_rank_correlation(x, y):
    correlation, p_value = stats.spearmanr(x, y)
    return {
        'test': "Spearman's Rank Correlation",
        'correlation': correlation,
        'p_value': p_value
    }

def chi_square_test(observed, expected):
    statistic, p_value = stats.chisquare(observed, expected)
    return {
        'test': 'Chi-Square Test',
        'statistic': statistic,
        'p_value': p_value
    }

def sign_test(data1, data2):
    statistic, p_value = stats.wilcoxon(data1, data2, alternative='two-sided', mode='approx', zero_method='wilcox')
    return {
        'test': 'Sign Test',
        'statistic': statistic,
        'p_value': p_value
    }

def kolmogorov_smirnov_test(data1, data2):
    statistic, p_value = stats.ks_2samp(data1, data2)
    return {
        'test': 'Kolmogorov-Smirnov Test',
        'statistic': statistic,
        'p_value': p_value
    }