import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import time
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import fsolve
try:
    from config import PRODUCTION_RATES, GITHUB_URL
    from utils import setup_logging, polynomial
    from random_point_generator import generate_df
    from validators import validate_conduit_size, validate_production_rate, get_valid_options, get_valid_glr_range, validate_glr, validate_positive_integer
except ImportError as e:
    st.error(f"Import error in ml_predictor: {str(e)}")
    raise

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = setup_logging()

def load_reference_data():
    """
    Load reference Excel file (referenceexcel.xlsx) from GitHub and parse into a list of dictionaries.
    Returns None if loading or parsing fails.
    """
    logger.info("Loading reference Excel file from GitHub...")
    try:
        response = requests.get(GITHUB_URL)
        response.raise_for_status()
        file_like_object = io.BytesIO(response.content)
        df_ref = pd.read_excel(file_like_object, header=None, engine='openpyxl')
        if df_ref.shape[1] < 6:
            st.error("Invalid Excel file: Must have at least 6 columns (name + 5 or 6 coefficients).")
            logger.error("Excel file has insufficient columns.")
            return None
        data_ref = []
        for index, row in df_ref.iterrows():
            name = row[0]
            if pd.isna(name) or isinstance(name, (int, float)):
                logger.warning(f"Skipping row {index} due to invalid name: {name}")
                continue
            name = str(name).strip()
            if not re.match(r'[\d.]+\s*in\s*\d+\s*stb-day\s*\d+\s*glr', name.lower()):
                logger.warning(f"Failed to parse reference data name: {name}")
                continue
            parts = name.split()
            try:
                conduit_size = float(parts[0])
                production_rate = float(parts[2])
                glr = float(parts[4].replace('glr', ''))
                coefficients = {
                    'a': float(row[1]),
                    'b': float(row[2]),
                    'c': float(row[3]),
                    'd': float(row[4]),
                    'e': float(row[5]),
                    'f': float(row[6]) if len(row) > 6 and pd.notna(row[6]) else 0.0
                }
                data_ref.append({
                    'conduit_size': conduit_size,
                    'production_rate': production_rate,
                    'glr': glr,
                    'coefficients': coefficients
                })
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing row {index}: {e}")
                continue
        if not data_ref:
            st.error("No valid data parsed from referenceexcel.xlsx.")
            logger.error("No valid data parsed from the Excel file.")
            return None
        logger.info("Reference data loaded successfully from referenceexcel.xlsx.")
        return data_ref
    except Exception as e:
        st.error(f"Error loading referenceexcel.xlsx from GitHub: {str(e)}")
        logger.error(f"Error loading reference Excel: {str(e)}")
        return None

def load_ml_data(reference_data, conduit_size, production_rate, num_points, glr=None, min_D=1000, all_prates=False, both_conduits=False):
    """
    Generate ML data using random_point_generator's generate_df function.
    Focuses on p1, D, p2 for prediction.
    """
    dfs_ml = []
    
    conduit_sizes = [2.875, 3.5] if both_conduits else [conduit_size]
    try:
        valid_prates_dict = {cs: [float(pr) for pr in get_valid_options(cs)[0]] for cs in conduit_sizes}
    except Exception as e:
        st.error(f"Error accessing valid options: {str(e)}")
        logger.error(f"Error in get_valid_options: {str(e)}")
        return None
    
    total_iterations = 0
    for cs in conduit_sizes:
        prates = valid_prates_dict.get(cs, []) if all_prates else [production_rate]
        for pr in prates:
            filtered_data = [
                entry for entry in reference_data 
                if entry['conduit_size'] == cs 
                and entry['production_rate'] == pr
                and (glr is None or entry['glr'] == glr)
            ]
            total_iterations += len(filtered_data) if glr is None else 1
    
    if total_iterations == 0:
        st.error("No data found for the selected parameters.")
        logger.error(f"No data for conduit_size={conduit_sizes}, production_rate={production_rate}, glr={glr}")
        return None
    
    progress_bar = st.progress(0)
    current_iteration = 0
    
    for cs in conduit_sizes:
        prates = valid_prates_dict.get(cs, []) if all_prates else [production_rate]
        for pr in prates:
            filtered_data = [
                entry for entry in reference_data 
                if entry['conduit_size'] == cs 
                and entry['production_rate'] == pr
                and (glr is None or entry['glr'] == glr)
            ]
            if not filtered_data:
                current_iteration += 1 if glr is not None else 1
                continue
            for entry in filtered_data:
                coeffs = [entry['coefficients'][k] for k in sorted(entry['coefficients'].keys())]
                try:
                    df_temp = generate_df(coeffs, num_points, min_D)
                except Exception as e:
                    logger.error(f"Error in generate_df: {str(e)}")
                    current_iteration += 1
                    continue
                if df_temp is None or df_temp.empty:
                    current_iteration += 1
                    continue
                if not all(col in df_temp.columns for col in ['p1', 'D', 'p2']):
                    logger.error(f"Generated DataFrame missing required columns: {df_temp.columns}")
                    current_iteration += 1
                    continue
                df_temp = df_temp[['p1', 'D', 'p2']].copy()
                df_temp['conduit_size'] = entry['conduit_size']
                df_temp['production_rate'] = entry['production_rate']
                df_temp['GLR'] = entry['glr']
                dfs_ml.append(df_temp)
                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations)
    
    if not dfs_ml:
        st.error("No valid data generated.")
        logger.error("No valid data generated.")
        return None
    
    df_ml = pd.concat(dfs_ml, ignore_index=True)
    logger.info(f"Generated ML data with {len(df_ml)} points across {len(dfs_ml)} configurations")
    return df_ml

def train_model(df_ml, model_type):
    """
    Train the selected ML model.
    """
    if df_ml.empty or df_ml is None:
        st.error("No data available for training.")
        return None, None
    features = ['p1', 'D', 'conduit_size', 'production_rate', 'GLR']
    target = 'p2'
    if not all(col in df_ml.columns for col in features + [target]):
        st.error(f"DataFrame missing required columns: {df_ml.columns}")
        logger.error(f"DataFrame missing required columns: {df_ml.columns}")
        return None, None
    X = df_ml[features]
    y = df_ml[target]
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
    except Exception as e:
        st.error(f"Error scaling data: {str(e)}")
        logger.error(f"Error scaling data: {str(e)}")
        return None, None
    
    st.write(f"Training {model_type} model...")
    progress = st.progress(0)
    
    try:
        if model_type == "Neural Network":
            model = Sequential([
                Input(shape=(X.shape[1],)),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            epochs = 50
            for epoch in range(epochs):
                model.fit(X_scaled, y, epochs=1, batch_size=32, validation_split=0.2, verbose=0)
                progress.progress((epoch + 1) / epochs)
                time.sleep(0.05)
        
        elif model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            progress.progress(1.0)
        
        elif model_type == "Gradient Boosting":
            model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            progress.progress(1.0)
        
        elif model_type == "Stacking Ensemble":
            estimators = [
                ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
                ('lgb', lgb.LGBMRegressor(n_estimators=50, random_state=42))
            ]
            model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
            model.fit(X_scaled, y)
            progress.progress(1.0)
        
        return model, scaler
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        logger.error(f"Training error: {str(e)}")
        return None, None

def predict_p2(model, scaler, p1, D, conduit_size, production_rate, glr):
    """
    Predict p2 using the trained model.
    """
    features = ['p1', 'D', 'conduit_size', 'production_rate', 'GLR']
    input_data = pd.DataFrame([{
        'p1': p1, 'D': D, 'conduit_size': conduit_size,
        'production_rate': production_rate, 'GLR': glr
    }])
    try:
        input_scaled = scaler.transform(input_data[features])
        prediction = model.predict(input_scaled)[0]
        if isinstance(prediction, np.ndarray):
            prediction = prediction.item()
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        logger.error(f"Prediction error: {str(e)}")
        return None

def calculate_p2_finder(reference_data, p1, D, conduit_size, production_rate, glr):
    """
    Calculate p2 using p2 Finder logic (polynomial-based from ui.py).
    """
    try:
        # Find or interpolate coefficients
        filtered_data = [
            entry for entry in reference_data
            if entry['conduit_size'] == conduit_size and entry['production_rate'] == production_rate
        ]
        if not filtered_data:
            logger.error(f"No data for conduit_size={conduit_size}, production_rate={production_rate}")
            return None
        glrs = [entry['glr'] for entry in filtered_data]
        if glr in glrs:
            coeffs = next(entry['coefficients'] for entry in filtered_data if entry['glr'] == glr)
        else:
            glrs.sort()
            if glr < glrs[0] or glr > glrs[-1]:
                logger.error(f"GLR {glr} out of range for conduit_size={conduit_size}, production_rate={production_rate}")
                return None
            lower_glr = max([g for g in glrs if g <= glr], default=None)
            upper_glr = min([g for g in glrs if g >= glr], default=None)
            if lower_glr is None or upper_glr is None:
                logger.error(f"Cannot interpolate GLR {glr}")
                return None
            lower_coeffs = next(entry['coefficients'] for entry in filtered_data if entry['glr'] == lower_glr)
            upper_coeffs = next(entry['coefficients'] for entry in filtered_data if entry['glr'] == upper_glr)
            weight = (glr - lower_glr) / (upper_glr - lower_glr)
            coeffs = {key: lower_coeffs[key] + (upper_coeffs[key] - lower_coeffs[key]) * weight
                      for key in lower_coeffs}
        
        # Calculate y1 = polynomial(p1)
        y1 = polynomial(p1, coeffs)
        if not np.isfinite(y1) or y1 < 0 or y1 > 31000:
            logger.error(f"Invalid y1={y1} for p1={p1}")
            return None
        
        # Calculate y2 = y1 + D
        y2 = y1 + D
        if y2 > 31000:
            logger.error(f"y2={y2} exceeds max depth 31000")
            return None
        
        # Solve polynomial(p2) = y2 for p2
        def root_function(p2):
            return polynomial(p2, coeffs) - y2
        
        p2_guess = p1 + 100
        p2 = fsolve(root_function, p2_guess, maxfev=20000)[0]
        if not np.isfinite(p2) or p2 < 0 or p2 > 4000:
            logger.error(f"Invalid p2={p2}")
            return None
        
        return p2
    except Exception as e:
        logger.error(f"Error in p2 Finder calculation: {str(e)}")
        return None

def run_ml_predictor():
    """
    UI for Bottomhole Pressure Predictor.
    """
    st.subheader("Mode 6: Bottomhole Pressure Predictor")
    
    # Apply theme from ui.py
    theme = 'plotly_white'
    if st.session_state.get('theme', 'light') == 'dark':
        theme = 'plotly_dark'
        st.markdown("""
            <style>
                .stApp {
                    background-color: #1e1e1e;
                    color: #ffffff;
                }
                .stTextInput > div > div > input, .stSelectbox > div > div > select {
                    background-color: #333333;
                    color: #ffffff;
                }
                .stButton > button {
                    background-color: #4CAF50;
                    color: white;
                }
            </style>
        """, unsafe_allow_html=True)
    
    # Load reference data if not already loaded
    if 'reference_data' not in st.session_state:
        with st.spinner("Loading reference data..."):
            reference_data = load_reference_data()
            if reference_data is None:
                st.error("Failed to load reference data.")
                return
            st.session_state.reference_data = reference_data
    else:
        reference_data = st.session_state.reference_data
    
    # Training data inputs
    st.subheader("Input Parameters for Training Data")
    col1, col2 = st.columns(2)
    
    with col1:
        conduit_size = st.selectbox("Conduit Size (in):", [2.875, 3.5], key="ml_pred_conduit")
        both_conduits = st.checkbox("Use Both Conduit Sizes", value=False, key="ml_pred_both_conduits")
    
    with col2:
        try:
            valid_prates, valid_glrs = get_valid_options(conduit_size)
            valid_prates = [float(pr) for pr in valid_prates]
        except Exception as e:
            st.error(f"Error loading valid options: {str(e)}")
            logger.error(f"Error loading valid options: {str(e)}")
            return
        production_rate = st.selectbox("Production Rate (stb/day):", valid_prates, key="ml_pred_prod_rate")
        all_prates = st.checkbox("Use All Production Rates", value=False, key="ml_pred_all_prates")
    
    num_points = st.number_input("Number of Random Points per GLR Curve:", min_value=1, value=1000, step=100, key="ml_pred_num_points")
    
    all_glr = st.checkbox("Use All GLRs for Selected Production Rate", value=True, key="ml_pred_all_glr")
    glr = None
    if not all_glr:
        valid_glrs_list = valid_glrs.get(production_rate, [])
        if valid_glrs_list:
            glr = st.selectbox("GLR (scf/stb):", [float(g) for g in valid_glrs_list], key="ml_pred_glr")
        else:
            st.error(f"No valid GLRs available for production rate {production_rate}.")
            return
    
    if st.button("Generate Data", key="ml_pred_generate"):
        errors = []
        if not both_conduits and not validate_conduit_size(conduit_size):
            errors.append("Invalid conduit size.")
        if not all_prates and not validate_production_rate(production_rate):
            errors.append("Invalid production rate.")
        if not validate_positive_integer(num_points, "number of random points"):
            errors.append("Invalid number of random points.")
        if not all_glr and glr is not None and not validate_glr(conduit_size, production_rate, glr):
            valid_range = get_valid_glr_range(conduit_size, production_rate)
            errors.append(f"Invalid GLR {glr}. Valid ranges: {valid_range}")
        
        if errors:
            for error in errors:
                st.error(error)
            logger.error(f"Errors in data generation: {errors}")
            return
        
        with st.spinner("Generating data..."):
            df_ml = load_ml_data(reference_data, conduit_size, production_rate, num_points, glr, 
                               all_prates=all_prates, both_conduits=both_conduits)
            if df_ml is None or df_ml.empty:
                st.error("Failed to generate data.")
                return
            st.session_state.df_ml_pred = df_ml
            st.subheader("Generated Data Preview")
            st.dataframe(df_ml.head())
            st.success("Data generation complete!")
    
    # Model training
    if 'df_ml_pred' in st.session_state:
        model_type = st.selectbox("Select ML Model:", 
                                ["Neural Network", "Random Forest", "Gradient Boosting", "Stacking Ensemble"], 
                                key="ml_pred_model_type")
        if st.button("Train Model", key="ml_pred_train"):
            model, scaler = train_model(st.session_state.df_ml_pred, model_type)
            if model is None:
                st.error("Training failed.")
                return
            st.session_state.model_pred = model
            st.session_state.scaler_pred = scaler
            st.success("Training complete!")
    
    # Prediction UI
    if 'model_pred' in st.session_state:
        st.subheader("Predict Bottomhole Flowing Pressure (p2)")
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            pred_p1 = st.number_input("Wellhead Pressure (p1, psi):", min_value=0.0, max_value=4000.0, value=1000.0, key="ml_pred_p1")
            pred_D = st.number_input("Well Length (D, ft):", min_value=0.0, max_value=31000.0, value=1000.0, key="ml_pred_D")
        
        with pred_col2:
            pred_conduit = st.selectbox("Conduit Size (in):", [2.875, 3.5], key="ml_pred_conduit_pred")
            pred_prod = st.selectbox("Production Rate (stb/day):", PRODUCTION_RATES, key="ml_pred_prod_pred")
            pred_glr = st.number_input("GLR (scf/stb):", min_value=0.0, max_value=25000.0, value=200.0, key="ml_pred_glr_pred")
        
        if st.button("Predict p2", key="ml_pred_predict"):
            errors = []
            if not validate_conduit_size(pred_conduit):
                errors.append("Invalid conduit size for prediction.")
            if not validate_production_rate(pred_prod):
                errors.append("Invalid production rate for prediction.")
            if not validate_glr(pred_conduit, pred_prod, pred_glr):
                valid_range = get_valid_glr_range(pred_conduit, pred_prod)
                errors.append(f"Invalid GLR {pred_glr}. Valid ranges: {valid_range}")
            
            if errors:
                for error in errors:
                    st.error(error)
                logger.error(f"Prediction errors: {errors}")
                return
            
            # ML Prediction
            p2_pred = predict_p2(st.session_state.model_pred, st.session_state.scaler_pred, 
                               pred_p1, pred_D, pred_conduit, pred_prod, pred_glr)
            
            # p2 Finder Calculation
            p2_finder = calculate_p2_finder(st.session_state.reference_data, pred_p1, pred_D,
                                          pred_conduit, pred_prod, pred_glr)
            
            if p2_pred is not None:
                st.success(f"ML Predicted p2: {p2_pred:.2f} psi")
            else:
                st.error("ML prediction failed.")
            
            if p2_finder is not None:
                st.success(f"p2 Finder Calculated p2: {p2_finder:.2f} psi")
            else:
                st.error("p2 Finder calculation failed.")
            
            # Plot comparison if both predictions are valid
            if p2_pred is not None and p2_finder is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[pred_p1], y=[p2_pred], mode='markers', name='ML Prediction',
                    marker=dict(size=12, color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=[pred_p1], y=[p2_finder], mode='markers', name='p2 Finder',
                    marker=dict(size=12, color='red')
                ))
                fig.update_layout(
                    title="p2 Comparison: ML Prediction vs p2 Finder",
                    xaxis_title="Wellhead Pressure (p1, psi)",
                    yaxis_title="Bottomhole Pressure (p2, psi)",
                    template=theme,
                    showlegend=True,
                    width=600,
                    height=400
                )
                st.plotly_chart(fig)
    else:
        st.warning("Please generate data and train the model before making predictions.")
