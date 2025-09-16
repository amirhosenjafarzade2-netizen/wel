import streamlit as st
import numpy as np
import pandas as pd
import xlsxwriter
import zipfile
import io
from scipy.optimize import fsolve, bisect
from data_loader import load_reference_data
from config import PRODUCTION_RATES, INTERPOLATION_RANGES
from utils import setup_logging
from validators import validate_conduit_size, validate_production_rate, get_valid_options, get_valid_glr_range, validate_glr, validate_positive_integer, validate_num_graph_sheets, validate_depth_and_pressure
import re

logger = setup_logging()

def parse_cell(cell):
    """
    Parse a cell reference (e.g., 'A1') into row and column indices.
    Returns (row, col) or raises ValueError.
    """
    try:
        col = ord(cell[0].upper()) - ord('A')
        row = int(cell[1:]) - 1
        if row < 0 or col < 0:
            raise ValueError("Row and column indices must be positive.")
        return row, col
    except (ValueError, IndexError):
        raise ValueError(f"Invalid cell reference format: {cell}. Use format like A1.")

def calc_y1(p, coeffs):
    """
    Calculate y1 (wellhead depth) using polynomial coefficients.
    Returns y1 (float) or None if calculation fails.
    """
    try:
        y = 0
        for i, coef in enumerate(coeffs):
            y += coef * p ** (len(coeffs) - 1 - i)
        if not np.isfinite(y):
            return None
        return y
    except Exception:
        return None

def solve_p2(y2_val, p1, coeffs):
    """
    Solve for p2 (bottomhole pressure) given y2, p1, and coefficients.
    Returns p2 (float) or None if calculation fails.
    """
    def polynomial(x, coeffs):
        try:
            y = 0
            for i, coef in enumerate(coeffs):
                y += coef * x ** (len(coeffs) - 1 - i)
            return y
        except Exception:
            return np.nan

    def root_function(x, target_depth, coeffs):
        return polynomial(x, coeffs) - target_depth

    p2 = None
    p_range = np.linspace(p1, 4000, 100)
    for i in range(len(p_range) - 1):
        p_start, p_end = p_range[i], p_range[i + 1]
        try:
            y_start = polynomial(p_start, coeffs)
            y_end = polynomial(p_end, coeffs)
            if not (np.isfinite(y_start) and np.isfinite(y_end)):
                continue
            f_start = root_function(p_start, y2_val, coeffs)
            f_end = root_function(p_end, y2_val, coeffs)
            if np.isfinite(f_start) and np.isfinite(f_end) and f_start * f_end <= 0:
                try:
                    candidate = fsolve(root_function, (p_start + p_end) / 2, args=(y2_val, coeffs), maxfev=20000)[0]
                    if p_start - 1e-6 <= candidate <= p_end + 1e-6:
                        y_candidate = polynomial(candidate, coeffs)
                        if np.isfinite(y_candidate) and 0 <= y_candidate <= 31000:
                            p2 = candidate
                            break
                except Exception:
                    pass
                try:
                    candidate = bisect(root_function, p_start, p_end, args=(y2_val, coeffs), maxiter=100)
                    y_candidate = polynomial(candidate, coeffs)
                    if np.isfinite(y_candidate) and 0 <= y_candidate <= 31000:
                        p2 = candidate
                        break
                except Exception:
                    continue
        except Exception:
            continue
    return p2

def generate_df(coeffs, num_points, min_D):
    """
    Generate DataFrame of random points for a given set of coefficients.
    Returns df or None if generation fails.
    """
    rows = []
    used_D_set = set()
    x_range = (0, 4000)
    y_range = (0, 31000)
    max_attempts = 50000
    attempts = 0

    while len(rows) < num_points and attempts < max_attempts:
        attempts += 1
        p1 = np.random.uniform(x_range[0], x_range[1])
        y1 = calc_y1(p1, coeffs)
        if y1 is None or not np.isfinite(y1) or y1 < 0 or y1 > y_range[1]:
            continue
        max_D = max(min_D, min(7000, y_range[1] - y1))
        if max_D < min_D:
            continue

        D_candidates = np.linspace(min_D, max_D, 1000)
        np.random.shuffle(D_candidates)
        D = None
        for d in D_candidates:
            if round(d, 8) not in used_D_set:
                D = d
                break
        if D is None:
            continue

        y2 = y1 + D
        if y2 > y_range[1]:
            y2 = y_range[1]
            D = y2 - y1

        p2 = solve_p2(y2, p1, coeffs)
        if p2 is None or p2 < x_range[0] or p2 > x_range[1]:
            continue

        used_D_set.add(round(D, 8))
        rows.append([p1, D, y1, y2, p2])

    df = pd.DataFrame(rows, columns=["p1", "D", "y1", "y2", "p2"])
    if df.empty:
        logger.error("No valid points generated")
        return None
    logger.info(f"Generated {len(df)} points for DataFrame")
    return df

def generate_excel(entry, num_points, min_D, generate_graphs, num_graph_sheets):
    """
    Generate an Excel file with random points and optional graph sheets.
    Returns (buffer, file_name) or (None, None) if generation fails.
    """
    try:
        df = generate_df([entry['coefficients'][k] for k in sorted(entry['coefficients'].keys())], num_points, min_D)
        if df is None or df.empty:
            logger.error(f"Failed to generate DataFrame for GLR {entry['glr']}")
            return None, None

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            if generate_graphs:
                workbook = writer.book
                for i in range(num_graph_sheets):
                    sheet_name = f'Graph_{i+1}'
                    worksheet = workbook.add_worksheet(sheet_name)
                    chart = workbook.add_chart({'type': 'scatter'})
                    chart.add_series({
                        'name': 'Pressure vs Depth',
                        'categories': ['Data', 1, 3, num_points, 3],  # y1 column
                        'values': ['Data', 1, 0, num_points, 0],      # p1 column
                    })
                    chart.set_x_axis({'name': 'Depth (ft)', 'reverse': True})
                    chart.set_y_axis({'name': 'Pressure (psi)'})
                    worksheet.insert_chart('A1', chart)
        
        file_name = f"random_points_{entry['conduit_size']}_in_{entry['production_rate']}_stb-day_{entry['glr']}_glr.xlsx"
        buffer.seek(0)
        logger.info(f"Generated Excel file: {file_name}")
        return buffer, file_name
    except Exception as e:
        logger.error(f"Excel generation failed for GLR {entry['glr']}: {str(e)}")
        return None, None

def interpolate_coefficients(reference_data, conduit_size, production_rate, glr):
    """
    Interpolate coefficients for a custom GLR between the nearest GLRs in reference_data.
    Returns a dictionary with interpolated coefficients or None if interpolation fails.
    """
    try:
        # Filter data for the given conduit size and production rate
        filtered_data = [entry for entry in reference_data 
                        if entry['conduit_size'] == conduit_size and entry['production_rate'] == production_rate]
        if not filtered_data:
            return None
        
        # Get all GLRs for this conduit size and production rate
        glrs = sorted([entry['glr'] for entry in filtered_data])
        if not glrs:
            return None
        
        # Find the nearest lower and higher GLRs
        lower_glr = max([g for g in glrs if g <= glr], default=None)
        higher_glr = min([g for g in glrs if g >= glr], default=None)
        
        if lower_glr is None or higher_glr is None:
            return None
        
        # If exact GLR match, return its coefficients
        if lower_glr == glr:
            entry = next(e for e in filtered_data if e['glr'] == glr)
            return entry['coefficients']
        
        # Get entries for lower and higher GLRs
        lower_entry = next(e for e in filtered_data if e['glr'] == lower_glr)
        higher_entry = next(e for e in filtered_data if e['glr'] == higher_glr)
        
        # Interpolate coefficients
        weight = (glr - lower_glr) / (higher_glr - lower_glr)
        interpolated_coeffs = {}
        for key in lower_entry['coefficients']:
            lower_coeff = lower_entry['coefficients'][key]
            higher_coeff = higher_entry['coefficients'][key]
            interpolated_coeffs[key] = lower_coeff + weight * (higher_coeff - lower_coeff)
        
        return interpolated_coeffs
    except Exception as e:
        logger.error(f"Failed to interpolate coefficients for GLR {glr}: {str(e)}")
        return None

def run_random_point_generator(reference_data, interpolation_ranges, production_rates):
    """
    UI for Random Point Generator: Generate random data points and output as Excel.
    """
    logger.info("Running Random Point Generator UI")
    
    if 'random_point_inputs' not in st.session_state:
        st.session_state.random_point_inputs = {
            'conduit_size': 2.875,
            'production_rate': 100.0,
            'glr': 200.0,
            'num_points': 100,
            'min_D': 1000.0,
            'generate_graphs': False,
            'num_graph_sheets': 1,
            'all_glr': True
        }
    
    st.subheader("Random Point Generator Inputs")
    col1, col2 = st.columns(2)
    
    with col1:
        valid_conduits = [2.875, 3.5]
        conduit_size = st.selectbox(
            "Conduit Size (in):",
            valid_conduits,
            index=valid_conduits.index(st.session_state.random_point_inputs['conduit_size']),
            key="random_conduit",
            help="Select the conduit size (2.875 or 3.5 inches)."
        )
        st.session_state.random_point_inputs['conduit_size'] = conduit_size
        
        valid_prates, valid_glrs = get_valid_options(conduit_size)
        valid_prates = [float(pr) for pr in valid_prates]
        production_rate = st.selectbox(
            "Production Rate (stb/day):",
            valid_prates,
            index=valid_prates.index(st.session_state.random_point_inputs['production_rate']) if st.session_state.random_point_inputs['production_rate'] in valid_prates else 0,
            key="random_prate",
            help="Select the production rate (50 to 600 stb/day)."
        )
        st.session_state.random_point_inputs['production_rate'] = production_rate
    
    with col2:
        num_points = st.number_input(
            "Number of Random Points:",
            min_value=1,
            value=int(st.session_state.random_point_inputs['num_points']),
            step=100,
            help="Number of random data points to generate."
        )
        st.session_state.random_point_inputs['num_points'] = num_points
        
        min_D = st.number_input(
            "Minimum Well Length, D (ft):",
            min_value=0.0,
            max_value=31000.0,
            value=float(st.session_state.random_point_inputs['min_D']),
            step=100.0,
            help="Minimum well length (y1 + D ≤ 31000 ft)."
        )
        st.session_state.random_point_inputs['min_D'] = min_D
    
    all_glr = st.checkbox(
        "Generate for All GLRs",
        value=st.session_state.random_point_inputs['all_glr'],
        help="Check to generate Excel files for all valid GLRs (up to 200 files in a ZIP)."
    )
    st.session_state.random_point_inputs['all_glr'] = all_glr
    
    glr = None
    if not all_glr:
        valid_glrs_list = [float(g) for g in valid_glrs.get(production_rate, [])]
        if valid_glrs_list:
            glr = st.selectbox(
                "GLR (scf/stb):",
                valid_glrs_list,
                index=valid_glrs_list.index(st.session_state.random_point_inputs['glr']) if st.session_state.random_point_inputs['glr'] in valid_glrs_list else 0,
                key="random_glr",
                help="Select GLR from valid values for the selected conduit size and production rate."
            )
            st.session_state.random_point_inputs['glr'] = glr
        else:
            st.error(f"No valid GLRs available for conduit size {conduit_size} and production rate {production_rate}.")
            logger.error(f"No valid GLRs for conduit size {conduit_size} and production rate {production_rate}.")
            return
    
    generate_graphs = st.checkbox(
        "Generate graph sheets in Excel?",
        value=st.session_state.random_point_inputs['generate_graphs'],
        help="Check to include graph sheets in the output Excel file(s)."
    )
    st.session_state.random_point_inputs['generate_graphs'] = generate_graphs
    
    num_graph_sheets = 0
    if generate_graphs:
        num_graph_sheets = st.number_input(
            "How many graph sheets? (1–10)",
            min_value=1,
            max_value=10,
            value=st.session_state.random_point_inputs['num_graph_sheets'],
            step=1,
            help="Number of graph sheets to include in each Excel file."
        )
        st.session_state.random_point_inputs['num_graph_sheets'] = num_graph_sheets
    
    generate = st.button("Generate Random Points")
    
    if generate:
        errors = []
        if not validate_conduit_size(conduit_size):
            errors.append("Invalid conduit size.")
        if not validate_production_rate(production_rate):
            errors.append("Invalid production rate.")
        if not validate_positive_integer(num_points, "number of random points"):
            errors.append("Invalid number of random points.")
        if not validate_depth_and_pressure(0, min_D):  # Check min_D against max_depth
            errors.append("Invalid minimum well length.")
        if generate_graphs and not validate_num_graph_sheets(num_graph_sheets):
            errors.append("Invalid number of graph sheets.")
        if not all_glr and glr is not None and not validate_glr(conduit_size, production_rate, glr):
            valid_range = get_valid_glr_range(conduit_size, production_rate)
            errors.append(f"Invalid GLR {glr}. Valid ranges: {valid_range}")
        
        if errors:
            for error in errors:
                st.error(error)
            logger.error(f"Random Point Generator errors: {errors}")
            return
        
        progress_bar = st.progress(0)
        try:
            filtered_data = [entry for entry in reference_data 
                            if entry['conduit_size'] == conduit_size and entry['production_rate'] == production_rate]
            if not filtered_data:
                st.error(f"No data found for conduit size {conduit_size} and production rate {production_rate} in reference data.")
                logger.error(f"No data found for conduit size {conduit_size} and production_rate {production_rate}.")
                return

            if all_glr:
                zip_buffer = io.BytesIO()
                total_glrs = len(filtered_data)
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for i, entry in enumerate(filtered_data):
                        excel_buffer, file_name = generate_excel(entry, num_points, min_D, generate_graphs, num_graph_sheets)
                        if excel_buffer and file_name:
                            zipf.writestr(file_name, excel_buffer.getvalue())
                        progress_bar.progress((i + 1) / total_glrs)
                zip_buffer.seek(0)
                st.download_button(
                    label="Download ZIP of All GLR Excels",
                    data=zip_buffer,
                    file_name="all_glr_random_points.zip",
                    mime="application/zip"
                )
            else:
                # Try to find exact GLR match
                entry = next((e for e in filtered_data if e['glr'] == glr), None)
                if entry is None:
                    # Interpolate coefficients for selected GLR
                    coefficients = interpolate_coefficients(reference_data, conduit_size, production_rate, glr)
                    if coefficients is None:
                        st.error(f"No data found for GLR {glr}. Valid ranges: {get_valid_glr_range(conduit_size, production_rate)}")
                        logger.error(f"No data found or interpolation failed for GLR {glr}.")
                        return
                    entry = {
                        'conduit_size': conduit_size,
                        'production_rate': production_rate,
                        'glr': glr,
                        'coefficients': coefficients
                    }
                df = generate_df([entry['coefficients'][k] for k in sorted(entry['coefficients'].keys())], num_points, min_D)
                if df is None or df.empty:
                    st.error(f"No valid points generated for GLR {glr}. Valid ranges: {get_valid_glr_range(conduit_size, production_rate)}")
                    logger.error(f"No valid points generated for GLR {glr}.")
                    return
                df['conduit_size'] = conduit_size
                df['production_rate'] = production_rate
                df['glr'] = glr
                st.subheader("Generated Data")
                st.dataframe(df)
                excel_buffer, file_name = generate_excel(entry, num_points, min_D, generate_graphs, num_graph_sheets)
                if excel_buffer and file_name:
                    st.download_button(
                        label="Download Excel",
                        data=excel_buffer,
                        file_name=file_name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                progress_bar.progress(1.0)
            
            st.success("Generation complete!")
        
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            logger.error(f"Random point generation failed: {str(e)}")
    
    st.write("**Generation Logs**")
    st.write("Any warnings or informational messages will appear here.")
