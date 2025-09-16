import streamlit as st
from data_loader import load_reference_data
from utils import setup_logging
from config import INTERPOLATION_RANGES, PRODUCTION_RATES
from ui import run_p2_finder, run_natural_flow_finder, run_glr_graph_drawer
from random_point_generator import run_random_point_generator
from ml_module import run_machine_learning
from ml_predictor import run_ml_predictor

# Initialize logger
logger = setup_logging()

def main():
    """
    Main function to run the Streamlit application.
    Initializes reference data and handles mode selection.
    """
    st.title("Well Pressure and Depth Calculator")

    # Initialize session state for mode selection
    if "mode_select" not in st.session_state:
        st.session_state.mode_select = None

    # Load reference data into session state (only once)
    if "REFERENCE_DATA" not in st.session_state:
        logger.info("Loading reference data...")
        reference_data = load_reference_data()
        if reference_data is None or not reference_data:
            st.error("Failed to initialize application: Unable to load reference data. "
                     "Please check the Excel file or internet connection.")
            logger.error("Application initialization failed due to missing or invalid reference data.")
            return
        st.session_state.REFERENCE_DATA = reference_data
        logger.info("Reference data loaded successfully.")

    # Validate configuration
    if not INTERPOLATION_RANGES or not PRODUCTION_RATES:
        st.error("Configuration error: Interpolation ranges or production rates are missing.")
        logger.error("Invalid configuration: INTERPOLATION_RANGES or PRODUCTION_RATES is empty.")
        return

    # Mode selection
    st.header("Select Calculation Mode")
    mode_options = [
        "Bottomhole Pressure Predictor",
        "Natural Flow Finder",
        "GLR Graph Drawer",
        "Random Point Generator",
        "Machine Learning Analysis",
        "Machine Learning Bottomhole Pressure Predictor"
    ]
    previous_mode = st.session_state.mode_select
    mode = st.selectbox("Choose a mode:", mode_options, key="mode_select")

    # Clear session state for specific modes when switching
    if mode != previous_mode:
        logger.info(f"Mode changed from {previous_mode} to {mode}")
        if mode != "Bottomhole Pressure Predictor":
            st.session_state.pop("p2_finder_inputs", None)
            st.session_state.pop("p2_finder_results", None)
        if mode != "Natural Flow Finder":
            st.session_state.pop("natural_flow_inputs", None)
            st.session_state.pop("natural_flow_results", None)
        if mode != "GLR Graph Drawer":
            st.session_state.pop("glr_graph_inputs", None)
        if mode != "Random Point Generator":
            st.session_state.pop("random_point_inputs", None)
        if mode != "Machine Learning Analysis":
            st.session_state.pop("df_ml", None)
            st.session_state.pop("model", None)
            st.session_state.pop("scaler", None)
        if mode != "Machine Learning Bottomhole Pressure Predictor":
            st.session_state.pop("df_ml_pred", None)
            st.session_state.pop("model_pred", None)
            st.session_state.pop("scaler_pred", None)

    # Create tabs for cleaner layout
    tabs = st.tabs(["Calculation", "Help"])

    with tabs[0]:
        try:
            if mode == "Bottomhole Pressure Predictor":
                run_p2_finder(st.session_state.REFERENCE_DATA, INTERPOLATION_RANGES, PRODUCTION_RATES)
            elif mode == "Natural Flow Finder":
                run_natural_flow_finder(st.session_state.REFERENCE_DATA, INTERPOLATION_RANGES, PRODUCTION_RATES)
            elif mode == "GLR Graph Drawer":
                run_glr_graph_drawer(st.session_state.REFERENCE_DATA, INTERPOLATION_RANGES, PRODUCTION_RATES)
            elif mode == "Random Point Generator":
                run_random_point_generator(st.session_state.REFERENCE_DATA, INTERPOLATION_RANGES, PRODUCTION_RATES)
            elif mode == "Machine Learning Analysis":
                run_machine_learning()
            elif mode == "Machine Learning Bottomhole Pressure Predictor":
                run_ml_predictor()
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            logger.error(f"Error in {mode} mode: {str(e)}")

    with tabs[1]:
        st.write("### Help")
        st.write("""
        **Well Pressure and Depth Calculator**
        - **Bottomhole Pressure Predictor**: Calculate wellhead (p1, y1) and bottomhole (p2, y2) pressures and depths.
        - **Natural Flow Finder**: Find the natural flow rate by intersecting TPR and IPR curves.
          - For Fetkovich method, choose to enter C and n directly or calculate them from up to four test points (Q0, Pwf).
        - **GLR Graph Drawer**: Plot pressure vs. depth curves for all GLRs at a given conduit size and production rate.
        - **Random Point Generator**: Generate random data points (p1, D, y1, y2, p2) for specified conduit size, production rate, and GLR. Outputs Excel file(s) with optional graph sheets (1-10 sheets). Choose a single GLR for one Excel or all GLRs for a ZIP of up to 200 Excel files.
        - **Machine Learning Analysis**: Train a neural network on data generated by Random Point Generator to analyze parameter effects or optimize well conditions using a genetic algorithm. Select between Parameter Analysis or Optimize Conditions after training.
        - **Machine Learning Bottomhole Pressure Predictor**: Train an ML model on generated data to predict bottomhole flowing pressure (p2) from inputs: wellhead pressure (p1), well length (D), conduit size, production rate, and GLR. Selectable models: Neural Network, Random Forest, Gradient Boosting, Stacking Ensemble.
        
        **Inputs**:
        - Conduit size: 2.875 or 3.5 inches
        - Production rate: 50, 100, 200, 400, or 600 stb/day
        - GLR: Gas-Liquid Ratio (varies by conduit size and production rate)
        - Depth (D): Well length in feet (y1 + D ≤ 31000 ft)
        - Pressures: p1, pwh, pr (0 to 4000 psi)
        - Fetkovich: C (> 0), n (0 to 2), or test points (Q0 > 0, Pwf > 0)
        - Random Point Generator: Number of points, minimum well length (D), generate graph sheets (yes/no), number of graph sheets (1-10)
        - Machine Learning: Number of random points per GLR, number of generations for optimization, analysis type (Parameter Analysis or Optimize Conditions)
        
        **Contact**: For issues, check the GitHub repository or contact support.
        """)

if __name__ == "__main__":
    main()
