import io
import base64
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.stattools import durbin_watson  # Add this import

def scatter_plot(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title("Scatter Plot")
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return image_base64

def correlation_coefficient(x, y):
    r, p_value = stats.pearsonr(x, y)
    return {
        'test': "Pearson Correlation Coefficient",
        'r': r,
        'p_value': p_value
    }

def durbin_watson_test(residuals):
    statistic = durbin_watson(residuals)  # Update this line
    return {
        'test': "Durbin-Watson Test",
        'statistic': statistic,
        'independence': 1.5 < statistic < 2.5
    }