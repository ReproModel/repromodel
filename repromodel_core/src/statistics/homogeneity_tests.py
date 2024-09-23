import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def levene_test(*args):
    statistic, p_value = stats.levene(*args)
    return {
        'test': "Levene's Test",
        'statistic': statistic,
        'p_value': p_value,
        'equal_variances': p_value > 0.05
    }

def bartlett_test(*args):
    statistic, p_value = stats.bartlett(*args)
    return {
        'test': "Bartlett's Test",
        'statistic': statistic,
        'p_value': p_value,
        'equal_variances': p_value > 0.05
    }

def box_plot(*args, labels=None):
    fig, ax = plt.subplots()
    ax.boxplot(args)
    if labels:
        ax.set_xticklabels(labels)
    ax.set_title("Box Plot")
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return image_base64