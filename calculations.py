import streamlit as st
import numpy as np
from scipy.optimize import root_scalar, curve_fit
from scipy.interpolate import interp1d
from config import INTERPOLATION_RANGES, PRODUCTION_RATES
from utils import polynomial, setup_logging
from validators import validate_conduit_size, validate_production_rate, validate_glr

# Initialize logger
logger = setup_logging()

@st.cache_data
def calculate_results(conduit_size, production_rate, glr_input, p1, D, data_ref):
    """
    Calculate depths (y1, y2) and pressure (p2) based on polynomial interpolation.
    Returns (y1, y2, p2, coeffs, interpolation_status, glr1, glr2) or (None, ...) if invalid.
    """
    logger.info(f"calculate_results inputs: conduit_size={conduit_size}, production_rate={production_rate}, "
                f"glr_input={glr_input}, p1={p1}, D={D}")
    
    if not validate_conduit_size(conduit_size):
        logger.error(f"Invalid conduit size: {conduit_size}")
        return None, None, None, None, None, None, None
    if not validate_production_rate(production_rate):
        logger.error(f"Invalid production rate: {production_rate}")
        return None, None, None, None, None, None, None
    if not validate_glr(conduit_size, production_rate, glr_input):
        logger.error(f"Invalid GLR {glr_input} for conduit {conduit_size}, production {production_rate}")
        return None, None, None, None, None, None, None

    lower_prate = max([pr for pr in PRODUCTION_RATES if pr <= production_rate], default=50)
    higher_prate = min([pr for pr in PRODUCTION_RATES if pr >= production_rate], default=600)
    production_interpolation_status = "exact" if abs(lower_prate - higher_prate) < 1e-6 else "interpolated"
    prate1, prate2 = lower_prate, higher_prate
    logger.info(f"Production rates for interpolation: prate1={prate1}, prate2={prate2}, status={production_interpolation_status}")

    valid_glr1, valid_range1 = False, None
    ranges1 = INTERPOLATION_RANGES.get((conduit_size, prate1), [])
    for min_glr, max_glr in ranges1:
        if min_glr <= glr_input <= max_glr:
            valid_glr1, valid_range1 = True, (min_glr, max_glr)
            break
    if not valid_glr1:
        logger.error(f"GLR {glr_input} outside valid ranges {ranges1} for conduit {conduit_size}, production {prate1}")
        return None, None, None, None, None, None, None

    valid_glr2, valid_range2 = False, None
    ranges2 = INTERPOLATION_RANGES.get((conduit_size, prate2), [])
    for min_glr, max_glr in ranges2:
        if min_glr <= glr_input <= max_glr:
            valid_glr2, valid_range2 = True, (min_glr, max_glr)
            break
    if not valid_glr2:
        logger.error(f"GLR {glr_input} outside valid ranges {ranges2} for conduit {conduit_size}, production {prate2}")
        return None, None, None, None, None, None, None

    if valid_glr1 and (glr_input < valid_range1[0] * 1.05 or glr_input > valid_range1[1] * 0.95):
        logger.warning(f"GLR {glr_input} is near edge of range {valid_range1} for production {prate1}")
    if valid_glr2 and (glr_input < valid_range2[0] * 1.05 or glr_input > valid_range2[1] * 0.95):
        logger.warning(f"GLR {glr_input} is near edge of range {valid_range2} for production {prate2}")

    def get_coefficients(conduit_size, production_rate, glr_input, valid_range, data_ref):
        """
        Get polynomial coefficients by interpolating between GLR values.
        Returns (coeffs, glr1, glr2, status) or (None, ...) if invalid.
        """
        logger.info(f"get_coefficients: conduit_size={conduit_size}, production_rate={production_rate}, "
                    f"glr_input={glr_input}, valid_range={valid_range}")
        
        # Check for exact match
        for entry in data_ref:
            if (abs(entry['conduit_size'] - conduit_size) < 1e-6 and
                abs(entry['production_rate'] - production_rate) < 1e-6 and
                abs(entry['glr'] - glr_input) < 1e-6):
                logger.info(f"Exact match found: coefficients={entry['coefficients']}")
                return entry['coefficients'], glr_input, glr_input, "exact"

        # Find relevant data points for interpolation
        relevant_rows = [
            entry for entry in data_ref
            if (abs(entry['conduit_size'] - conduit_size) < 1e-6 and
                abs(entry['production_rate'] - production_rate) < 1e-6 and
                valid_range[0] <= entry['glr'] <= valid_range[1])
        ]
        if not relevant_rows:
            logger.error(f"No data points for conduit {conduit_size}, production {production_rate}, GLR range {valid_range}")
            return None, None, None, None

        relevant_rows.sort(key=lambda x: x['glr'])
        logger.debug(f"Relevant rows: {[r['glr'] for r in relevant_rows]}")

        if len(relevant_rows) == 1:
            if abs(relevant_rows[0]['glr'] - glr_input) < 1e-6:
                logger.info(f"Single exact match: coefficients={relevant_rows[0]['coefficients']}")
                return relevant_rows[0]['coefficients'], glr_input, glr_input, "exact"
            logger.warning(f"Only one GLR point ({relevant_rows[0]['glr']}) available, cannot interpolate")
            return None, None, None, None

        lower_row = max([r for r in relevant_rows if r['glr'] <= glr_input], key=lambda x: x['glr'], default=relevant_rows[0])
        higher_row = min([r for r in relevant_rows if r['glr'] >= glr_input], key=lambda x: x['glr'], default=relevant_rows[-1])
        glr1, glr2 = lower_row['glr'], higher_row['glr']
        logger.debug(f"Interpolation GLRs: glr1={glr1}, glr2={glr2}")

        if abs(glr1 - glr2) < 1e-6:
            logger.info(f"Using exact coefficients for glr={glr1}")
            return lower_row['coefficients'], glr1, glr2, "exact"

        fraction = (glr_input - glr1) / (glr2 - glr1)
        if not (0 <= fraction <= 1):
            logger.warning(f"Invalid interpolation fraction: {fraction}")
            return None, None, None, None

        coeffs = {
            'a': lower_row['coefficients']['a'] + fraction * (higher_row['coefficients']['a'] - lower_row['coefficients']['a']),
            'b': lower_row['coefficients']['b'] + fraction * (higher_row['coefficients']['b'] - lower_row['coefficients']['b']),
            'c': lower_row['coefficients']['c'] + fraction * (higher_row['coefficients']['c'] - lower_row['coefficients']['c']),
            'd': lower_row['coefficients']['d'] + fraction * (higher_row['coefficients']['d'] - lower_row['coefficients']['d']),
            'e': lower_row['coefficients']['e'] + fraction * (higher_row['coefficients']['e'] - lower_row['coefficients']['e']),
            'f': lower_row['coefficients']['f'] + fraction * (higher_row['coefficients']['f'] - lower_row['coefficients']['f'])
        }
        logger.info(f"Interpolated coefficients: {coeffs}")
        return coeffs, glr1, glr2, "interpolated"

    coeffs1, glr1_lower, glr1_higher, glr_status1 = get_coefficients(conduit_size, prate1, glr_input, valid_range1, data_ref)
    if coeffs1 is None:
        logger.error(f"Failed to get coefficients for prate1={prate1}")
        return None, None, None, None, None, None, None

    if production_interpolation_status == "exact":
        coeffs, glr1, glr2, interpolation_status = coeffs1, glr1_lower, glr1_higher, glr_status1
    else:
        coeffs2, glr2_lower, glr2_higher, glr_status2 = get_coefficients(conduit_size, prate2, glr_input, valid_range2, data_ref)
        if coeffs2 is None:
            logger.error(f"Failed to get coefficients for prate2={prate2}")
            return None, None, None, None, None, None, None
        fraction_prate = (production_rate - prate1) / (prate2 - prate1)
        if not (0 <= fraction_prate <= 1):
            logger.warning(f"Invalid production rate interpolation fraction: {fraction_prate}")
            return None, None, None, None, None, None, None
        coeffs = {
            'a': coeffs1['a'] + fraction_prate * (coeffs2['a'] - coeffs1['a']),
            'b': coeffs1['b'] + fraction_prate * (coeffs2['b'] - coeffs1['b']),
            'c': coeffs1['c'] + fraction_prate * (coeffs2['c'] - coeffs1['c']),
            'd': coeffs1['d'] + fraction_prate * (coeffs2['d'] - coeffs1['d']),
            'e': coeffs1['e'] + fraction_prate * (coeffs2['e'] - coeffs1['e']),
            'f': coeffs1['f'] + fraction_prate * (coeffs2['f'] - coeffs1['f'])
        }
        glr1 = glr1_lower if glr_status1 == "exact" else min(glr1_lower, glr1_higher)
        glr2 = glr2_higher if glr_status2 == "exact" else max(glr2_lower, glr2_higher)
        interpolation_status = "interpolated" if glr_status1 == "interpolated" or glr_status2 == "interpolated" else "exact"
        logger.info(f"Production rate interpolation: fraction={fraction_prate}, coeffs={coeffs}")

    y1 = polynomial(p1, coeffs)
    if not np.isfinite(y1) or y1 < 0 or y1 > 31000:
        logger.error(f"Computed y1 ({y1:.2f} ft) is invalid or outside range 0 to 31000 ft")
        return None, None, None, None, None, None, None

    y2 = y1 + D
    if y2 > 31000:
        y2 = 31000
        D_adjusted = y2 - y1
        logger.info(f"Adjusted y2 to {y2:.2f} ft, D_adjusted={D_adjusted:.2f} ft to stay within 31000 ft")

    def root_function(x, target_depth, coeffs):
        return polynomial(x, coeffs) - target_depth

    p2 = None
    try:
        result = root_scalar(
            root_function, args=(y2, coeffs),
            bracket=[0, 4000], method='brentq'
        )
        if result.converged:
            p2 = result.root
    except ValueError as e:
        logger.debug(f"Brentq failed for p2: {str(e)}")
        try:
            result = root_scalar(
                root_function, args=(y2, coeffs),
                x0=p1, x1=p1 + 100, method='secant'
            )
            if result.converged:
                p2 = result.root
        except ValueError as e:
            logger.error(f"No valid p2 found for y2={y2:.2f} ft: {str(e)}")
            return None, None, None, None, None, None, None

    if p2 is None or not np.isfinite(p2) or p2 < 0 or p2 > 4000:
        logger.error(f"Computed p2 ({p2 if p2 is not None else 'None'} psi) is invalid or outside range 0 to 4000 psi")
        return None, None, None, None, None, None, None

    logger.info(f"Calculated: y1={y1:.2f} ft, y2={y2:.2f} ft, p2={p2:.2f} psi, interpolation_status={interpolation_status}")
    return y1, y2, p2, coeffs, interpolation_status, glr1, glr2

@st.cache_data
def calculate_tpr_points(conduit_size, glr, D, pwh, data_ref):
    """
    Calculate TPR points for given conduit size, GLR, depth, and wellhead pressure.
    Returns list of (production_rate, p2) tuples.
    """
    logger.info(f"calculate_tpr_points inputs: conduit_size={conduit_size}, glr={glr}, D={D}, pwh={pwh}")
    if not validate_conduit_size(conduit_size):
        logger.error(f"Invalid conduit size: {conduit_size}")
        raise ValueError(f"Invalid conduit size: {conduit_size}")
    if not validate_glr(conduit_size, PRODUCTION_RATES[0], glr):
        logger.error(f"Invalid GLR {glr} for conduit size {conduit_size}")
        raise ValueError(f"Invalid GLR {glr}")
    if not np.isfinite(D) or D <= 0 or D > 31000:
        logger.error(f"Invalid depth: D={D}")
        raise ValueError(f"Invalid depth: {D}")
    if not np.isfinite(pwh) or pwh < 0 or pwh > 4000:
        logger.error(f"Invalid wellhead pressure: pwh={pwh}")
        raise ValueError(f"Invalid wellhead pressure: {pwh}")

    tpr_points = []
    for prate in PRODUCTION_RATES:
        try:
            result = calculate_results(conduit_size, prate, glr, pwh, D, data_ref)
            if result[0] is not None:
                y1, y2, p2, coeffs, interpolation_status, glr1, glr2 = result
                if np.isfinite(p2) and 0 <= p2 <= 4000:
                    tpr_points.append((prate, p2))
                    logger.info(f"Valid TPR point: (prate={prate}, p2={p2:.2f})")
                else:
                    logger.warning(f"Invalid p2 ({p2:.2f} psi) for production rate {prate} stb/day")
            else:
                logger.warning(f"Failed to compute p2 for production rate {prate} stb/day")
        except Exception as e:
            logger.warning(f"Error computing TPR point for production rate {prate}: {str(e)}")
            continue

    if not tpr_points:
        logger.error("No valid TPR points computed. Check data_ref or input parameters.")
        raise ValueError("No valid TPR points computed. Check data_ref or input parameters.")
    
    logger.info(f"Generated {len(tpr_points)} TPR points: {tpr_points}")
    return tpr_points

@st.cache_data
def calculate_ipr_fetkovich(pr, c=None, n=None, q01=None, pwf1=None, q02=None, pwf2=None, q03=None, pwf3=None, q04=None, pwf4=None):
    """
    Calculate IPR parameters and points using Fetkovich method.
    If c and n are provided, use them directly; otherwise, calculate from points.
    Returns (c, n, ipr_points, fetkovich_points).
    """
    logger.info(f"calculate_ipr_fetkovich inputs: pr={pr}, c={c}, n={n}, points=[{q01, pwf1}, {q02, pwf2}, {q03, pwf3}, {q04, pwf4}]")
    if not np.isfinite(pr) or pr <= 0 or pr > 10000:
        logger.error(f"Invalid reservoir pressure: pr={pr}")
        raise ValueError(f"Invalid reservoir pressure: {pr}")

    points = []
    for q, pwf in [(q01, pwf1), (q02, pwf2), (q03, pwf3), (q04, pwf4)]:
        if (q is not None and pwf is not None and 
            np.isfinite(q) and np.isfinite(pwf) and 
            q > 0 and 0 <= pwf <= pr):
            points.append((q, pwf))
        else:
            logger.warning(f"Skipping invalid Fetkovich point: q={q}, pwf={pwf}")

    if c is not None and n is not None:
        if c <= 0 or not np.isfinite(c) or n <= 0 or n > 2.0 or not np.isfinite(n):
            logger.error(f"Invalid Fetkovich parameters: c={c}, n={n}")
            raise ValueError(f"Invalid Fetkovich parameters: c={c}, n={n}")
    else:
        if len(points) < 2:
            logger.error(f"Insufficient valid points for Fetkovich calculation: {len(points)} points provided")
            raise ValueError("At least two valid points required for Fetkovich parameters")
        
        if len(points) == 2:
            q01, pwf1 = points[0]
            q02, pwf2 = points[1]
            if pwf1 == pwf2 or q01 == q02:
                logger.error("Invalid Fetkovich inputs: Pwf1, Pwf2, Q01, Q02 must be distinct")
                raise ValueError("Invalid Fetkovich input parameters")
            delta_p1 = pr**2 - pwf1**2
            delta_p2 = pr**2 - pwf2**2
            if delta_p1 <= 0 or delta_p2 <= 0 or delta_p1 == delta_p2:
                logger.error("Invalid delta pressures for Fetkovich calculation")
                raise ValueError("Invalid delta pressures")
            n = np.log10(q02 / q01) / np.log10(delta_p2 / delta_p1)
            c = q01 / (delta_p1 ** n)
        else:
            q_points_list, pwf_points_list = zip(*points)
            q_points_array = np.array(q_points_list, dtype=float)
            pwf_points_array = np.array(pwf_points_list, dtype=float)
            def fetkovich_model(pwf, c, n):
                return c * (pr**2 - pwf**2)**n
            try:
                popt, _ = curve_fit(fetkovich_model, pwf_points_array, q_points_array, p0=[1e-5, 0.5], maxfev=10000)
                c, n = popt
            except Exception as e:
                logger.error(f"Curve fit failed for Fetkovich: {str(e)}")
                raise ValueError("Curve fit failed")

        if c <= 0 or not np.isfinite(c) or n <= 0 or n > 2.0 or not np.isfinite(n):
            logger.error(f"Invalid Fetkovich parameters: c={c}, n={n}")
            raise ValueError(f"Invalid Fetkovich parameters: c={c}, n={n}")

    q_max = c * (pr**2) ** n  # Estimate maximum flow rate
    if not np.isfinite(q_max) or q_max <= 0:
        logger.error(f"Invalid q_max: {q_max}")
        raise ValueError(f"Invalid q_max: {q_max}")

    pwf_values = np.linspace(0, pr, 50)
    ipr_points = []
    for pwf in pwf_values:
        q0 = c * (pr**2 - pwf**2)**n
        if np.isfinite(q0) and 0 <= q0 <= q_max * 1.1:
            ipr_points.append((q0, pwf))
        else:
            logger.debug(f"Excluded invalid IPR point: q0={q0:.2f}, pwf={pwf:.2f}")
    
    if len(ipr_points) < 2:
        logger.error("Insufficient valid IPR points for Fetkovich")
        raise ValueError("Insufficient valid IPR points")

    logger.info(f"Fetkovich parameters: c={c:.4e}, n={n:.4f}, points={len(ipr_points)}, input_points={len(points)}")
    return c, n, ipr_points, points

@st.cache_data
def calculate_ipr_vogel(pr, q_max):
    """
    Calculate IPR points using Vogel method.
    Returns (q_max, ipr_points).
    """
    logger.info(f"calculate_ipr_vogel inputs: pr={pr}, q_max={q_max}")
    if not np.isfinite(pr) or pr <= 0 or pr > 10000:
        logger.error(f"Invalid reservoir pressure: pr={pr}")
        raise ValueError(f"Invalid reservoir pressure: {pr}")
    if not np.isfinite(q_max) or q_max <= 0:
        logger.error(f"Invalid q_max: {q_max}")
        raise ValueError(f"Invalid q_max: {q_max}")

    pwf_values = np.linspace(0, pr, 50)
    ipr_points = []
    for pwf in pwf_values:
        q0 = q_max * (1 - 0.2 * (pwf / pr) - 0.8 * (pwf / pr)**2)
        if np.isfinite(q0) and 0 <= q0 <= q_max * 1.1:
            ipr_points.append((q0, pwf))
        else:
            logger.debug(f"Excluded invalid IPR point: q0={q0:.2f}, pwf={pwf:.2f}")
    
    if len(ipr_points) < 2:
        logger.error("Insufficient valid IPR points for Vogel")
        raise ValueError("Insufficient valid IPR points")
    
    logger.info(f"Vogel parameters: q_max={q_max:.4f}, points={len(ipr_points)}")
    return q_max, ipr_points

@st.cache_data
def calculate_ipr_composite(pr, j_star, p_b):
    """
    Calculate IPR points using Composite method.
    Returns (j_star, p_b, ipr_points).
    """
    logger.info(f"calculate_ipr_composite inputs: pr={pr}, j_star={j_star}, p_b={p_b}")
    if not np.isfinite(pr) or pr <= 0 or pr > 10000:
        logger.error(f"Invalid reservoir pressure: pr={pr}")
        raise ValueError(f"Invalid reservoir pressure: {pr}")
    if not np.isfinite(j_star) or j_star <= 0:
        logger.error(f"Invalid j_star: {j_star}")
        raise ValueError(f"Invalid j_star: {j_star}")
    if not np.isfinite(p_b) or p_b <= 0 or p_b > pr:
        logger.error(f"Invalid bubble point pressure: p_b={p_b}")
        raise ValueError(f"Invalid bubble point pressure: {p_b}")

    pwf_values = np.linspace(0, pr, 50)
    ipr_points = []
    for pwf in pwf_values:
        if pwf > p_b:
            q0 = j_star * (pr - pwf)
        else:
            q0 = j_star * (pr - p_b) + (j_star * p_b / 1.8) * (1 - 0.2 * (pwf / p_b) - 0.8 * (pwf / p_b)**2)
        if np.isfinite(q0) and 0 <= q0 <= 1000:
            ipr_points.append((q0, pwf))
        else:
            logger.debug(f"Excluded invalid IPR point: q0={q0:.2f}, pwf={pwf:.2f}")
    
    if len(ipr_points) < 2:
        logger.error("Insufficient valid IPR points for Composite")
        raise ValueError("Insufficient valid IPR points")
    
    logger.info(f"Composite parameters: j_star={j_star:.4f}, p_b={p_b:.4f}, points={len(ipr_points)}")
    return j_star, p_b, ipr_points

@st.cache_data
def find_intersection(tpr_points, ipr_points, pr):
    """
    Find intersection between TPR and IPR curves using interpolation.
    Returns (intersection_q0, intersection_p) or (None, None) if no valid intersection.
    """
    logger.info(f"find_intersection inputs: tpr_points={tpr_points}, ipr_points={ipr_points}, pr={pr}")
    if not tpr_points or not ipr_points:
        logger.error("Empty TPR or IPR points provided")
        return None, None

    # Filter valid points
    tpr_points = [(q, p) for q, p in tpr_points if np.isfinite(q) and np.isfinite(p) and q >= 0 and p >= 0]
    ipr_points = [(q, p) for q, p in ipr_points if np.isfinite(q) and np.isfinite(p) and q >= 0 and p >= 0]
    
    if len(tpr_points) < 2 or len(ipr_points) < 2:
        logger.error(f"Insufficient points: TPR={len(tpr_points)}, IPR={len(ipr_points)}")
        return None, None

    try:
        tpr_q0, tpr_p2 = zip(*tpr_points)
        ipr_q0, ipr_pwf = zip(*ipr_points)
        
        tpr_q0 = np.array(tpr_q0, dtype=float)
        tpr_p2 = np.array(tpr_p2, dtype=float)
        ipr_q0 = np.array(ipr_q0, dtype=float)
        ipr_pwf = np.array(ipr_pwf, dtype=float)
        
        # Sort points by q0
        tpr_indices = np.argsort(tpr_q0)
        tpr_q0 = tpr_q0[tpr_indices]
        tpr_p2 = tpr_p2[tpr_indices]
        
        ipr_indices = np.argsort(ipr_q0)
        ipr_q0 = ipr_q0[ipr_indices]
        ipr_pwf = ipr_pwf[ipr_indices]

        logger.info(f"TPR points: {len(tpr_points)}, IPR points: {len(ipr_points)}")
        logger.debug(f"TPR q0: {tpr_q0.tolist()}, TPR p2: {tpr_p2.tolist()}")
        logger.debug(f"IPR q0: {ipr_q0.tolist()}, IPR pwf: {ipr_pwf.tolist()}")

        # Determine common q0 range
        q_min = max(min(tpr_q0), min(ipr_q0))
        q_max = min(max(tpr_q0), max(ipr_q0))
        
        if q_min >= q_max:
            logger.warning(f"No valid intersection range: q_min={q_min:.2f}, q_max={q_max:.2f}")
            return None, None

        # Create interpolation functions
        try:
            tpr_interp = interp1d(tpr_q0, tpr_p2, kind='linear', fill_value='extrapolate')
            ipr_interp = interp1d(ipr_q0, ipr_pwf, kind='linear', fill_value='extrapolate')
        except Exception as e:
            logger.error(f"Interpolation failed: {str(e)}")
            return None, None

        def diff_func(q):
            try:
                tpr_val = tpr_interp(q)
                ipr_val = ipr_interp(q)
                if not (np.isfinite(tpr_val) and np.isfinite(ipr_val)):
                    logger.debug(f"Non-finite interpolation at q={q}: tpr_val={tpr_val}, ipr_val={ipr_val}")
                    return np.inf
                return ipr_val - tpr_val
            except Exception as e:
                logger.debug(f"Intersection function error at q={q}: {str(e)}")
                return np.inf

        # Check for sign change to ensure intersection exists
        f_min = diff_func(q_min)
        f_max = diff_func(q_max)
        if not (np.isfinite(f_min) and np.isfinite(f_max)):
            logger.warning(f"Non-finite function values: f_min={f_min}, f_max={f_max}")
            return None, None

        if f_min * f_max >= 0:
            logger.warning("No sign change in interpolation function, no intersection found")
            return None, None

        # Find intersection
        try:
            result = root_scalar(diff_func, bracket=[q_min, q_max], method='brentq')
            if result.converged:
                intersection_q0 = result.root
                intersection_p = tpr_interp(intersection_q0)
                if (0 <= intersection_q0 <= 1000 and 0 <= intersection_p <= pr):
                    logger.info(f"Intersection found: q0={intersection_q0:.2f}, p={intersection_p:.2f}")
                    return intersection_q0, intersection_p
                else:
                    logger.warning(f"Intersection out of bounds: q0={intersection_q0:.2f}, p={intersection_p:.2f}")
                    return None, None
            else:
                logger.warning("Intersection calculation did not converge")
                return None, None
        except Exception as e:
            logger.warning(f"Intersection calculation failed: {str(e)}")
            return None, None
    except Exception as e:
        logger.error(f"Intersection calculation failed: {str(e)}")
        return None, None
