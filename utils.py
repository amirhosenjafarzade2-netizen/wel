import numpy as np
import pandas as pd
import io
import logging
import matplotlib.pyplot as plt

def polynomial(x, coeffs):
    """
    Calculate polynomial values for given input x and coefficients.
    Vectorized to handle arrays efficiently.
    """
    try:
        x = np.asarray(x)
        return (coeffs['a'] * x**5 +
                coeffs['b'] * x**4 +
                coeffs['c'] * x**3 +
                coeffs['d'] * x**2 +
                coeffs['e'] * x +
                coeffs['f'])
    except Exception:
        return np.nan

def setup_logging():
    """
    Set up logging configuration for debugging and application logs.
    Logs to both console and a file ('app.log').
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
    return logging.getLogger(__name__)

def export_plot_to_png(fig):
    """
    Export a matplotlib figure to PNG bytes for download.
    """
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"PNG export failed: {str(e)}")
        raise Exception(f"PNG export failed: {str(e)}")
