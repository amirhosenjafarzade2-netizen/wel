import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools
import random
import requests
import io
import re
import time
from config import INTERPOLATION_RANGES, PRODUCTION_RATES, GITHUB_URL
from utils import setup_logging
from random_point_generator import generate_df, calc_y1, solve_p2
from validators import validate_conduit_size, validate_production_rate, get_valid_options, get_valid_glr_range, validate_glr, validate_positive_integer

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
    If glr is None, use all valid GLRs. If all_prates is True, use all valid production rates.
    If both_conduits is True, use both conduit sizes (2.875 and 3.5).
    """
    dfs_ml = []
    required_cols = ["p1", "D", "y1", "y2", "p2"]
    
    conduit_sizes = [2.875, 3.5] if both_conduits else [conduit_size]
    valid_prates_dict = {cs: [float(pr) for pr in get_valid_options(cs)[0]] for cs in conduit_sizes}
    
    total_iterations = 0
    for cs in conduit_sizes:
        prates = valid_prates_dict[cs] if all_prates else [production_rate]
        for pr in prates:
            filtered_data = [
                entry for entry in reference_data 
                if entry['conduit_size'] == cs 
                and entry['production_rate'] == pr
                and (glr is None or entry['glr'] == glr)
            ]
            total_iterations += len(filtered_data) if glr is None else 1
    
    if total_iterations == 0:
        valid_range = get_valid_glr_range(conduit_size, production_rate)
        st.error(f"No data found for conduit size(s) {conduit_sizes}, production rate(s) {prates}" +
                 (f", GLR {glr}. Valid ranges: {valid_range}" if glr else f". Valid ranges: {valid_range}"))
        logger.error(f"No data found for conduit_size(s)={conduit_sizes}, production_rate(s)={prates}, glr={glr}")
        return None
    
    progress_bar = st.progress(0)
    current_iteration = 0
    
    for cs in conduit_sizes:
        prates = valid_prates_dict[cs] if all_prates else [production_rate]
        for pr in prates:
            filtered_data = [
                entry for entry in reference_data 
                if entry['conduit_size'] == cs 
                and entry['production_rate'] == pr
                and (glr is None or entry['glr'] == glr)
            ]
            if not filtered_data:
                valid_range = get_valid_glr_range(cs, pr)
                logger.warning(f"No data for conduit_size={cs}, production_rate={pr}, glr={glr}. Valid ranges: {valid_range}")
                current_iteration += 1 if glr is not None else len(get_valid_glr_range(cs, pr))
                continue
            for entry in filtered_data:
                coeffs = [entry['coefficients'][k] for k in sorted(entry['coefficients'].keys())]
                df_temp = generate_df(coeffs, num_points, min_D)
                if df_temp is None or df_temp.empty:
                    logger.warning(f"Failed to generate data for conduit={entry['conduit_size']}, prod={entry['production_rate']}, glr={entry['glr']}")
                    current_iteration += 1
                    continue
                df_temp['conduit_size'] = entry['conduit_size']
                df_temp['production_rate'] = entry['production_rate']
                df_temp['GLR'] = entry['glr']
                dfs_ml.append(df_temp)
                logger.info(f"Generated {len(df_temp)} points for conduit={cs}, production_rate={pr}, glr={entry['glr']}")
                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations)
    
    if not dfs_ml:
        valid_range = get_valid_glr_range(conduit_size, production_rate)
        st.error(f"No valid data generated for conduit size(s) {conduit_sizes}, production rate(s) {prates}" +
                 (f", GLR {glr}. Valid ranges: {valid_range}" if glr else f". Valid ranges: {valid_range}"))
        logger.error(f"No valid data generated for conduit_size(s)={conduit_sizes}, production_rate(s)={prates}, glr={glr}")
        return None
    
    df_ml = pd.concat(dfs_ml, ignore_index=True)
    logger.info(f"Generated ML data with {len(df_ml)} points across {len(dfs_ml)} configurations")
    return df_ml

def train_neural_network(df_ml):
    """
    Train the neural network with a progress bar for epochs.
    """
    if df_ml.empty:
        return None, None
    features = ['p1', 'D', 'y1', 'y2', 'conduit_size', 'production_rate', 'GLR']
    target = 'p2'
    X = df_ml[features]
    y = df_ml[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    st.write("Training neural network...")
    progress = st.progress(0)
    epochs = 50
    for epoch in range(epochs):
        model.fit(X_scaled, y, epochs=1, batch_size=32, validation_split=0.2, verbose=0)
        progress.progress((epoch + 1) / epochs)
        time.sleep(0.05)
    return model, scaler

def analyze_parameter_effects(model, scaler, df_ml):
    """
    Generate parameter effect plots matching the main Colab program.
    """
    features = ['p1', 'D', 'y1', 'y2', 'conduit_size', 'production_rate', 'GLR']
    for conduit_size in [2.875, 3.5]:
        for production_rate in PRODUCTION_RATES:
            glr_min, glr_max = get_valid_glr_range(conduit_size, production_rate)
            glr_values = np.linspace(glr_min, glr_max, 100)
            base_values = df_ml[features].mean().to_dict()
            base_values['conduit_size'] = conduit_size
            base_values['production_rate'] = production_rate
            X_test_glr = pd.DataFrame([base_values] * 100)
            X_test_glr['GLR'] = glr_values
            X_test_glr_scaled = scaler.transform(X_test_glr[features])
            glr_predictions = model.predict(X_test_glr_scaled, verbose=0).flatten()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(glr_values, glr_predictions, label=f'Conduit: {conduit_size} in, Prod: {production_rate} stb/day')
            ax.set_xlabel('GLR (SCF/STB)')
            ax.set_ylabel('Pressure Gradient (psi)')
            ax.set_title('Pressure Gradient vs. GLR')
            ax.grid(True)
            ax.legend()
            st.subheader(f"Effect of GLR (Conduit: {conduit_size} in, Production: {production_rate} stb/day)")
            st.pyplot(fig)

    glr_values = np.linspace(0, 25000, 100)
    base_values = df_ml[features].mean().to_dict()
    X_test_glr = pd.DataFrame([base_values] * 100)
    X_test_glr['GLR'] = glr_values
    X_test_glr_scaled = scaler.transform(X_test_glr[features])
    glr_predictions = model.predict(X_test_glr_scaled, verbose=0).flatten()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(glr_values, glr_predictions, label='Overall Pressure vs. GLR')
    ax.set_xlabel('GLR (SCF/STB)')
    ax.set_ylabel('Pressure Gradient (psi)')
    ax.set_title('Overall Pressure Gradient vs. GLR')
    ax.grid(True)
    ax.legend()
    st.subheader("Overall Pressure vs. GLR")
    st.pyplot(fig)

    for param in ['D', 'production_rate', 'conduit_size']:
        if param == 'conduit_size':
            param_values = [2.875, 3.5]
        elif param == 'production_rate':
            param_values = PRODUCTION_RATES
        else:
            param_values = np.linspace(df_ml[param].min(), df_ml[param].max(), 100)
        base_values = df_ml[features].mean().to_dict()
        X_test_param = pd.DataFrame([base_values] * len(param_values))
        X_test_param[param] = param_values
        X_test_param_scaled = scaler.transform(X_test_param[features])
        param_predictions = model.predict(X_test_param_scaled, verbose=0).flatten()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(param_values, param_predictions, label=f'Pressure Gradient vs. {param}')
        ax.set_xlabel(param.capitalize())
        ax.set_ylabel('Pressure Gradient (psi)')
        ax.set_title(f'Pressure Gradient vs. {param.capitalize()}')
        ax.grid(True)
        ax.legend()
        st.subheader(f"Pressure Gradient vs. {param.capitalize()}")
        st.pyplot(fig)

def get_valid_glr_range(conduit_size, production_rate):
    ranges = INTERPOLATION_RANGES.get((conduit_size, production_rate), [])
    return min(r[0] for r in ranges) if ranges else 0, max(r[1] for r in ranges) if ranges else 25000

def get_valid_glrs(reference_data, conduit_size, production_rate):
    """
    Get the list of valid discrete GLR values for the given conduit_size and production_rate.
    """
    return sorted([
        entry['glr'] for entry in reference_data
        if entry['conduit_size'] == conduit_size and entry['production_rate'] == production_rate
    ])

def evaluate_individual(individual, model, scaler):
    features = ['p1', 'D', 'y1', 'y2', 'conduit_size', 'production_rate', 'GLR']
    conduit_size, production_rate, glr = individual
    input_data = pd.DataFrame([{
        'p1': 1000, 'D': 1000, 'y1': 5000, 'y2': 6000,
        'conduit_size': conduit_size, 'production_rate': production_rate, 'GLR': glr
    }])
    input_scaled = scaler.transform(input_data[features])
    prediction = model.predict(input_scaled, verbose=0)[0][0]
    return (prediction,)

def custom_mutation(individual, indpb, conduit_sizes, production_rates, valid_glrs):
    """
    Custom mutation to ensure valid conduit_size, production_rate, and GLR.
    """
    if random.random() < indpb:
        individual[0] = random.choice(conduit_sizes)
    if random.random() < indpb:
        individual[1] = random.choice(production_rates)
    if random.random() < indpb and valid_glrs:
        individual[2] = random.choice(valid_glrs)
    return individual,

def optimize_neural_network_conditions(model, scaler, df_ml, reference_data, n_generations=20):
    """
    Optimize neural network conditions using a genetic algorithm with constrained values.
    """
    logger.info(f"Starting optimization with {n_generations} generations")
    features = ['p1', 'D', 'y1', 'y2', 'conduit_size', 'production_rate', 'GLR']
    conduit_sizes = [2.875, 3.5]
    production_rates = PRODUCTION_RATES

    if hasattr(creator, 'FitnessMin'):
        del creator.FitnessMin
    if hasattr(creator, 'Individual'):
        del creator.Individual
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_conduit", random.choice, conduit_sizes)
    toolbox.register("attr_prod", random.choice, production_rates)
    def attr_glr():
        conduit = random.choice(conduit_sizes)
        prod = random.choice(production_rates)
        valid_glrs = get_valid_glrs(reference_data, conduit, prod)
        return random.choice(valid_glrs) if valid_glrs else random.uniform(0, 25000)
    toolbox.register("attr_glr", attr_glr)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_conduit, toolbox.attr_prod, toolbox.attr_glr), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual, model=model, scaler=scaler)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutation, indpb=0.2, conduit_sizes=conduit_sizes,
                     production_rates=production_rates, valid_glrs=[])
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=50)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    best_fitness_history = []
    output_container = st.empty()
    progress = output_container.progress(0)
    for gen in range(n_generations):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                child1[0] = min(conduit_sizes, key=lambda x: abs(x - child1[0]))
                child1[1] = min(production_rates, key=lambda x: abs(x - child1[1]))
                valid_glrs = get_valid_glrs(reference_data, child1[0], child1[1])
                child1[2] = min(valid_glrs, key=lambda x: abs(x - child1[2])) if valid_glrs else child1[2]
                child2[0] = min(conduit_sizes, key=lambda x: abs(x - child2[0]))
                child2[1] = min(production_rates, key=lambda x: abs(x - child2[1]))
                valid_glrs = get_valid_glrs(reference_data, child2[0], child2[1])
                child2[2] = min(valid_glrs, key=lambda x: abs(x - child2[2])) if valid_glrs else child2[2]
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            valid_glrs = get_valid_glrs(reference_data, mutant[0], mutant[1])
            toolbox.mutate(mutant, valid_glrs=valid_glrs)
            del mutant.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
        best_ind = tools.selBest(pop, 1)[0]
        best_fitness_history.append(best_ind.fitness.values[0])
        progress.progress((gen + 1) / n_generations)
        logger.info(f"Generation {gen + 1}/{n_generations}: Best fitness = {best_ind.fitness.values[0]:.2f}")
    
    output_container.empty()
    st.subheader("Optimal Conditions for Minimal Pressure Gradient")
    st.write(f"Conduit Size: {best_ind[0]} in")
    st.write(f"Production Rate: {best_ind[1]} stb/day")
    st.write(f"GLR: {best_ind[2]:.2f} SCF/STB")
    st.write(f"Predicted Pressure Gradient: {best_fitness_history[-1]:.2f} psi")
    logger.info(f"Optimization complete: Conduit {best_ind[0]}, Production {best_ind[1]}, GLR {best_ind[2]:.2f}, Fitness {best_fitness_history[-1]:.2f}")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, n_generations + 1), best_fitness_history, marker='o')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Pressure Gradient (psi)')
    ax.set_title('Optimization Progress: Best Fitness per Generation')
    ax.grid(True)
    st.subheader("Optimization Progress: Best Fitness per Generation")
    st.pyplot(fig)
    
    base_values = df_ml[features].mean().to_dict()
    base_values['conduit_size'] = best_ind[0]
    base_values['production_rate'] = best_ind[1]
    base_values['GLR'] = best_ind[2]
    
    production_rates = PRODUCTION_RATES
    X_test_prod = pd.DataFrame([base_values] * len(production_rates))
    X_test_prod['production_rate'] = production_rates
    X_test_prod_scaled = scaler.transform(X_test_prod[features])
    prod_predictions = model.predict(X_test_prod_scaled, verbose=0).flatten()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(production_rates, prod_predictions, marker='o', label='Pressure Gradient vs. Production Rate')
    ax.set_xlabel('Production Rate (stb/day)')
    ax.set_ylabel('Pressure Gradient (psi)')
    ax.set_title('Pressure Gradient vs. Production Rate')
    ax.grid(True)
    ax.legend()
    st.subheader("Pressure Gradient vs. Production Rate (Optimization)")
    st.pyplot(fig)
    
    glr_min, glr_max = get_valid_glr_range(best_ind[0], best_ind[1])
    glr_values = np.linspace(glr_min, glr_max, 100)
    X_test_glr = pd.DataFrame([base_values] * 100)
    X_test_glr['GLR'] = glr_values
    X_test_glr_scaled = scaler.transform(X_test_glr[features])
    glr_predictions = model.predict(X_test_glr_scaled, verbose=0).flatten()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(glr_values, glr_predictions, label='Pressure Gradient vs. GLR')
    ax.set_xlabel('GLR (SCF/STB)')
    ax.set_ylabel('Pressure Gradient (psi)')
    ax.set_title('Pressure Gradient vs. GLR')
    ax.grid(True)
    ax.legend()
    st.subheader("Pressure Gradient vs. GLR (Optimization)")
    st.pyplot(fig)
    
    depth_values = np.linspace(df_ml['D'].min(), df_ml['D'].max(), 100)
    X_test_depth = pd.DataFrame([base_values] * 100)
    X_test_depth['D'] = depth_values
    X_test_depth_scaled = scaler.transform(X_test_depth[features])
    depth_predictions = model.predict(X_test_depth_scaled, verbose=0).flatten()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depth_values, depth_predictions, label='Pressure Gradient vs. Depth')
    ax.set_xlabel('Depth Offset (ft)')
    ax.set_ylabel('Pressure Gradient (psi)')
    ax.set_title('Pressure Gradient vs. Depth')
    ax.grid(True)
    ax.legend()
    st.subheader("Pressure Gradient vs. Depth (Optimization)")
    st.pyplot(fig)

def run_machine_learning():
    """
    UI for Machine Learning Analysis: Train neural network and analyze/optimize conditions.
    """
    st.subheader("Mode 5: Machine Learning Analysis")
    
    if 'reference_data' not in st.session_state:
        with st.spinner("Loading referenceexcel.xlsx from GitHub..."):
            reference_data = load_reference_data()
            if reference_data is None:
                st.error("Failed to load referenceexcel.xlsx.")
                return
            st.session_state.reference_data = reference_data
    else:
        reference_data = st.session_state.reference_data
    
    st.subheader("Input Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        conduit_size = st.selectbox(
            "Conduit Size (in):",
            [2.875, 3.5],
            help="Select the conduit size (2.875 or 3.5 inches)."
        )
        both_conduits = st.checkbox(
            "Use Both Conduit Sizes",
            value=False,
            help="Check to generate data for both conduit sizes (2.875 and 3.5 inches)."
        )
    
    with col2:
        valid_prates, valid_glrs = get_valid_options(conduit_size)
        valid_prates = [float(pr) for pr in valid_prates]
        production_rate = st.selectbox(
            "Production Rate (stb/day):",
            valid_prates,
            help="Select the production rate (50 to 600 stb/day)."
        )
        all_prates = st.checkbox(
            "Use All Production Rates",
            value=False,
            help="Check to generate data for all valid production rates for the selected conduit size(s)."
        )
    
    num_points = st.number_input(
        "Number of Random Points per GLR Curve:",
        min_value=1,
        value=1000,
        step=100,
        help="Number of random data points to generate per GLR curve."
    )
    
    n_generations = st.number_input(
        "Number of Generations for Optimization:",
        min_value=1,
        value=20,
        step=1,
        help="Number of generations for the genetic algorithm (used in Optimization)."
    )
    
    all_glr = st.checkbox("Use All GLRs for Selected Production Rate", value=True)
    glr = None
    
    if not all_glr:
        valid_glrs = valid_glrs.get(production_rate, [])
        if valid_glrs:
            glr = st.selectbox(
                "GLR (scf/stb):",
                [float(g) for g in valid_glrs],
                help="Select a valid GLR for the chosen conduit size and production rate."
            )
    
    if st.button("Generate Data and Train"):
        errors = []
        if not both_conduits and not validate_conduit_size(conduit_size):
            errors.append("Invalid conduit size.")
        if not all_prates and not validate_production_rate(production_rate):
            errors.append("Invalid production rate.")
        if not validate_positive_integer(num_points, "number of random points"):
            errors.append("Invalid number of random points.")
        if not validate_positive_integer(n_generations, "number of generations"):
            errors.append("Invalid number of generations.")
        if not all_glr and glr is not None and not validate_glr(conduit_size, production_rate, glr):
            valid_range = get_valid_glr_range(conduit_size, production_rate)
            errors.append(f"Invalid GLR {glr}. Valid ranges: {valid_range}")
        
        if errors:
            for error in errors:
                st.error(error)
            logger.error(f"Machine Learning Analysis errors: {errors}")
            return
        
        with st.spinner("Generating data..."):
            df_ml = load_ml_data(st.session_state.reference_data, conduit_size, production_rate, num_points, glr, all_prates=all_prates, both_conduits=both_conduits)
            if df_ml is None or df_ml.empty:
                valid_range = get_valid_glr_range(conduit_size, production_rate)
                st.error(f"Failed to generate ML data. Valid ranges: {valid_range}")
                return
            st.session_state.df_ml = df_ml
            st.subheader("Generated Data Preview")
            st.dataframe(df_ml.head())
        
        try:
            model, scaler = train_neural_network(st.session_state.df_ml)
            if model is None:
                st.error("Training failed.")
                return
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.success("Training complete!")
        except Exception as e:
            st.error(f"Error in Machine Learning mode: {str(e)}")
            logger.error(f"Error in Machine Learning mode: {str(e)}")
    
    if 'model' in st.session_state and 'reference_data' in st.session_state:
        option = st.selectbox(
            "Choose Analysis Type:",
            ["Neural Network", "Optimization"],
            key="analysis_type_selectbox"
        )
        logger.info(f"Selected analysis type: {option}")
        try:
            if option == "Neural Network":
                logger.info("Running parameter analysis")
                analyze_parameter_effects(st.session_state.model, st.session_state.scaler, st.session_state.df_ml)
            elif option == "Optimization":
                logger.info("Running optimization")
                optimize_neural_network_conditions(st.session_state.model, st.session_state.scaler, st.session_state.df_ml, st.session_state.reference_data, n_generations)
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            logger.error(f"Error in analysis: {str(e)}")
    else:
        st.warning("Please generate data and train the model before selecting an analysis type.")
