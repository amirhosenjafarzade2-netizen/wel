# config.py
# Centralized configuration for constants used in the Well Pressure and Depth Calculator

# Interpolation ranges for conduit size and production rate
INTERPOLATION_RANGES = {
    (2.875, 50): [(0, 10000), (10000, 17500)],
    (2.875, 100): [(0, 10000), (10000, 12500)],
    (2.875, 200): [(0, 6000), (6000, 8000)],
    (2.875, 400): [(0, 4000), (4000, 6500)],
    (2.875, 600): [(0, 3000), (3000, 5000)],
    (3.5, 50): [(0, 15000), (15000, 25000)],
    (3.5, 100): [(0, 10000), (10000, 17500)],
    (3.5, 200): [(0, 8000), (8000, 12000)],
    (3.5, 400): [(0, 8000), (8000, 9000)],
    (3.5, 600): [(0, 4000), (4000, 6000)]
}

# Available production rates for interpolation
PRODUCTION_RATES = [50, 100, 200, 400, 600]

# Define a larger set of distinct colors for colorful graphs
COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
    '#aec7e8',  # Light Blue
    '#ffbb78',  # Light Orange
    '#98df8a',  # Light Green
    '#ff9896',  # Light Red
    '#c5b0d5',  # Light Purple
]

# Map GLR values to specific colors for consistency across graphs
GLR_COLOR_MAP = {}

# GitHub URL for reference Excel file
GITHUB_URL = "https://raw.githubusercontent.com/amirhosenjafarzade2-netizen/well-flow-calculator/main/referenceexcel.xlsx"
