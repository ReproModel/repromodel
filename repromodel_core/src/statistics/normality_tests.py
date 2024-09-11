import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI plotting
import matplotlib.pyplot as plt
from statistics import NormalDist  # Add this line
import seaborn as sns
import io
import base64

def shapiro_wilk_test(data, alpha=0.05):
    statistic, p_value = stats.shapiro(data)
    return {
        'test': 'Shapiro-Wilk',
        'statistic': statistic,
        'p_value': p_value,
        'is_normal': p_value > alpha
    }

def kolmogorov_smirnov_test(data, alpha=0.05):
    statistic, p_value = stats.kstest(data, 'norm')
    return {
        'test': 'Kolmogorov-Smirnov',
        'statistic': statistic,
        'p_value': p_value,
        'is_normal': p_value > alpha
    }

def anderson_darling_test(data, alpha=0.05):
    result = stats.anderson(data)
    critical_values = result.critical_values
    significance_levels = [15, 10, 5, 2.5, 1]
    
    is_normal = result.statistic < critical_values[significance_levels.index(alpha * 100)]
    
    return {
        'test': 'Anderson-Darling',
        'statistic': result.statistic,
        'critical_values': dict(zip(significance_levels, critical_values)),
        'is_normal': is_normal
    }

def qq_plot(data):
    fig, ax = plt.subplots()
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot")
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image_base64

def histogram(data):
    fig, ax = plt.subplots()
    sns.histplot(data, kde=True, ax=ax)
    ax.set_title("Histogram with KDE")
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image_base64