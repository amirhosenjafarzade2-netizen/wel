# data_loader.py
# Functions to load and parse reference Excel data for the Well Pressure and Depth Calculator

import streamlit as st
import pandas as pd
import requests
import io
import re
from config import GITHUB_URL
from utils import setup_logging

# Initialize logger
logger = setup_logging()

@st.cache_data
def load_reference_data():
    """
    Load reference Excel file from GitHub and parse into a list of dictionaries.
    Returns None if loading or parsing fails.
    """
    logger.info("Loading reference Excel file from GitHub...")
    try:
        # Fetch the file from GitHub
        response = requests.get(GITHUB_URL)
        response.raise_for_status()  # Check for HTTP errors
        # Convert the response content to a file-like object
        file_like_object = io.BytesIO(response.content)
        df_ref = pd.read_excel(file_like_object, header=None, engine='openpyxl')
        # Validate file structure
        if df_ref.shape[1] < 6:
            st.error("Invalid Excel file: Must have at least 6 columns (name + 5 or 6 coefficients).")
            logger.error("Excel file has insufficient columns.")
            return None
        data_ref = []
        for index, row in df_ref.iterrows():
            name = row[0]
            if pd.isna(name) or isinstance(name, (int, float)):
                logger.warning(f"Skipping row {index} due to invalid name: {name}")
                st.warning(f"Skipping row {index} due to invalid name. Please ensure names are valid strings.")
                continue
            name = str(name).strip()
            if not re.match(r'[\d.]+\s*in\s*\d+\s*stb-day\s*\d+\s*glr', name.lower()):
                logger.warning(f"Failed to parse reference data name: {name}")
                st.warning(f"Invalid name format in row {index}: {name}. Expected format: 'X in Y stb-day Z glr'.")
                continue
            conduit_size, production_rate, glr = parse_name(name)
            if conduit_size is None:
                logger.error(f"Failed to parse reference data name: {name}")
                st.error(f"Error parsing name in row {index}: {name}. Please check the format.")
                continue
            try:
                coefficients = {
                    'a': float(row[1]),
                    'b': float(row[2]),
                    'c': float(row[3]),
                    'd': float(row[4]),
                    'e': float(row[5]),
                    'f': float(row[6]) if len(row) > 6 and pd.notna(row[6]) else 0.0
                }
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing coefficients in row {index}: {e}")
                st.error(f"Error parsing coefficients in row {index}: {e}. Please ensure all coefficients are numeric.")
                continue
            data_ref.append({
                'conduit_size': conduit_size,
                'production_rate': production_rate,
                'glr': glr,
                'coefficients': coefficients
            })
        if not data_ref:
            logger.error("No valid data parsed from the reference Excel file.")
            st.error("No valid data parsed from the Excel file. Please check the file content and format.")
            return None
        logger.info("Reference data loaded successfully from GitHub.")
        st.success("Reference data loaded successfully.")
        return data_ref
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading Excel file from GitHub: {str(e)}")
        st.error(f"Failed to download reference data from GitHub: {str(e)}. Please check the URL or your internet connection.")
        return None
    except Exception as e:
        logger.error(f"Error loading reference Excel file: {str(e)}")
        st.error(f"Error loading reference data: {str(e)}. Please ensure the file is a valid Excel file.")
        return None

def parse_name(name):
    """
    Parse the name column from the reference Excel file to extract conduit size, production rate, and GLR.
    Returns (conduit_size, production_rate, glr) or (None, None, None) if parsing fails.
    """
    try:
        parts = name.split()
        conduit_size = float(parts[0])
        production_rate = float(parts[2])
        glr_str = parts[4].replace('glr', '')
        glr = float(glr_str)
        return conduit_size, production_rate, glr
    except (IndexError, ValueError):
        logger.error(f"Failed to parse reference data name: {name}")
        return None, None, None
