"""Medical Insurance Cost Prediction - Interactive Streamlit Web Application.

Provides a clean, intuitive frontend for estimating annual healthcare insurance charges
based on individual demographic and biometric attributes. Uses the serialized
scikit-learn Linear Regression pipeline directly without duplicating preprocessing logic.
"""

import os
from pathlib import Path
import sys
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

# Safe import for environments where streamlit is installed or being tested
try:
    import streamlit as st
except ImportError:
    st = None

# Ensure project root is in sys.path so custom classes like ClinicalFeatureEngineer resolve
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import custom classes before loading the serialized pipeline.  Joblib records
# their module paths, so these definitions must remain available at runtime.
try:
    import src.data_preprocessing  # noqa: F401
    import src.train_model  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "The project's src package is required to load the trained model. "
        "Run Streamlit from the project root after installing requirements."
    ) from exc

DEFAULT_MODEL_PATH = "models/insurance_model.joblib"
MANDATORY_DISCLAIMER = "Disclaimer: This tool provides estimated charges for educational/informational purposes only and is not an actual insurance quote."

BENCHMARK_MEDIAN_CHARGE = 9382.03
BENCHMARK_MEAN_CHARGE = 13270.42


def get_model_path(model_path: str = DEFAULT_MODEL_PATH) -> Path:
    """Resolve model path relative to project root or current working directory."""
    path = Path(model_path)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load the trained Linear Regression pipeline artifact.

    Parameters
    ----------
    model_path : str
        Relative or absolute filesystem path to the serialized joblib pipeline.

    Returns
    -------
    Pipeline
        The loaded scikit-learn Pipeline object.
    """
    resolved_path = get_model_path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at '{resolved_path}'. "
            "Run 'python -m src.train_model' from the project root to generate it."
        )
    return joblib.load(str(resolved_path))


# Cache model loading when Streamlit runtime is active
if st is not None and hasattr(st, "cache_resource"):
    load_model = st.cache_resource(load_model)


def predict_insurance_cost(
    model,
    age: int,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
) -> float:
    """Compute estimated medical insurance charges from raw features.

    Parameters
    ----------
    model : Pipeline
        Serialized scikit-learn pipeline.
    age : int
        Age in years.
    sex : str
        Biological sex ('female' or 'male').
    bmi : float
        Body Mass Index (kg/m²).
    children : int
        Number of dependent children.
    smoker : str
        Smoking status ('no' or 'yes').
    region : str
        Residential quadrant ('northeast', 'northwest', 'southeast', 'southwest').

    Returns
    -------
    float
        Estimated annual charges in USD (guarded non-negative).
    """
    # Create raw DataFrame strictly with canonical 6 features - NO duplicate preprocessing
    raw_input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }])

    # Directly invoke pipeline's predict method
    raw_prediction = model.predict(raw_input_df)

    # Guard against theoretical negative OLS boundary extrapolation
    return max(0.0, float(raw_prediction[0]))


def assess_risk_profile(
    age: int,
    bmi: float,
    smoker: str,
) -> Dict[str, str]:
    """Generate risk assessment indicators for contextual health insights.

    Parameters
    ----------
    age : int
        Insured age.
    bmi : float
        Body Mass Index.
    smoker : str
        Smoking status.

    Returns
    -------
    dict
        Dictionary containing category ratings and contextual commentary.
    """
    if bmi < 18.5:
        bmi_cat = "Underweight (< 18.5)"
        bmi_status = "info"
    elif bmi < 25.0:
        bmi_cat = "Normal weight (18.5 – 24.9)"
        bmi_status = "success"
    elif bmi < 30.0:
        bmi_cat = "Overweight (25.0 – 29.9)"
        bmi_status = "warning"
    else:
        bmi_cat = "Obese (≥ 30.0)"
        bmi_status = "danger"

    is_smoker = (smoker.strip().lower() == "yes")
    is_obese = (bmi >= 30.0)

    if is_smoker and is_obese:
        risk_level = "High Clinical Risk (Compound Obese Smoker Tier)"
        risk_color = "#E53E3E"
        risk_summary = (
            "Tobacco smoking combined with clinical obesity triggers the primary non-linear "
            "cost escalation tier, significantly increasing anticipated medical expenditures."
        )
    elif is_smoker:
        risk_level = "Elevated Risk (Smoker Tier)"
        risk_color = "#DD6B20"
        risk_summary = (
            "Active smoking is the single largest isolated contributor to elevated annual "
            "actuarial charges."
        )
    elif is_obese:
        risk_level = "Moderate Risk (Non-Smoker, Obese)"
        risk_color = "#D69E2E"
        risk_summary = (
            "Higher BMI contributes moderately to expected costs, but remains substantially "
            "lower without tobacco smoking."
        )
    else:
        risk_level = "Standard Baseline Risk (Low Risk Profile)"
        risk_color = "#38A169"
        risk_summary = (
            "Standard demographic baseline risk tier. Medical charges scale primarily "
            "with policyholder age and number of covered dependents."
        )

    return {
        "bmi_category": bmi_cat,
        "bmi_status": bmi_status,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_summary": risk_summary,
    }


def main():
    """Build and render the Streamlit user interface."""
    if st is None:
        print("Error: Streamlit is not installed in the current Python environment.", file=sys.stderr)
        print("Please install Streamlit using: pip install streamlit", file=sys.stderr)
        print("Then launch the app with: streamlit run app.py", file=sys.stderr)
        sys.exit(1)

    # 1. Page Configuration & Meta
    st.set_page_config(
        page_title="Medical Insurance Cost Predictor",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 2. Custom CSS for Clean, Professional UI Styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1A365D;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #4A5568;
            margin-bottom: 1.2rem;
        }
        .prediction-card {
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            border-radius: 12px;
            padding: 24px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .prediction-label {
            font-size: 1.0rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            opacity: 0.9;
            margin-bottom: 8px;
        }
        .prediction-value {
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .disclaimer-container {
            background-color: #FFFBEB;
            border-left: 4px solid #F59E0B;
            padding: 16px 20px;
            border-radius: 6px;
            margin-top: 28px;
            color: #92400E;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .profile-summary-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
        }
        .profile-summary-table td {
            padding: 6px 12px;
            border-bottom: 1px solid #E2E8F0;
        }
        .profile-summary-table td.label {
            font-weight: 600;
            color: #4A5568;
            width: 45%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 3. Header Section
    st.markdown('<div class="main-header">🏥 Medical Insurance Cost Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">An Interactive Machine Learning Tool for Estimating Annual Healthcare Expenditures</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "This interactive application estimates annual medical insurance charges using an "
        "interpretable Ordinary Least Squares (OLS) Linear Regression model trained on US demographic "
        "and health risk metrics. Provide the individual profile attributes below to obtain a real-time prediction."
    )
    st.divider()

    # 4. Model Loading with User Feedback
    try:
        model = load_model()
    except Exception as err:
        st.error(f"⚠️ Model Loading Error: {err}")
        st.info(
            "Verify that the project contains the 'src' folder and that "
            "'models/insurance_model.joblib' exists. If the model file is missing, run "
            "'python -m src.train_model' from the project root."
        )
        return

    # 5. Two-Column Layout for Inputs and Prediction
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.subheader("📋 Policyholder Profile")

        # Demographic Attributes Sub-section
        st.markdown("##### Demographic & Family Factors")
        age = st.slider(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=30,
            step=1,
            help="Age of the primary insured policyholder (18–100).",
        )

        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            sex = st.radio(
                "Biological Sex",
                options=["female", "male"],
                index=0,
                horizontal=True,
                help="Biological sex assigned at birth.",
            )
        with sub_col2:
            children = st.slider(
                "Dependent Children",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                help="Number of dependent children covered under the health plan.",
            )

        st.markdown("##### Clinical & Health Risk Factors")
        bmi = st.slider(
            "Body Mass Index (BMI)",
            min_value=10.0,
            max_value=60.0,
            value=28.0,
            step=0.1,
            format="%.1f",
            help="Body Mass Index in kg/m² (Normal: 18.5–24.9, Overweight: 25.0–29.9, Obese: ≥30.0).",
        )

        sub_col3, sub_col4 = st.columns(2)
        with sub_col3:
            smoker = st.radio(
                "Smoking Status",
                options=["no", "yes"],
                index=0,
                horizontal=True,
                help="Active tobacco smoking status ('yes' or 'no').",
            )
        with sub_col4:
            region = st.selectbox(
                "Residential Region",
                options=["northeast", "northwest", "southeast", "southwest"],
                index=0,
                help="Beneficiary's residential geographic quadrant within the United States.",
            )

        calc_button = st.button("🔄 Recalculate Charges", type="primary", use_container_width=True)

    # 6. Real-Time Model Inference & Risk Assessment
    prediction = predict_insurance_cost(
        model=model,
        age=age,
        sex=sex,
        bmi=bmi,
        children=children,
        smoker=smoker,
        region=region,
    )
    risk_info = assess_risk_profile(age=age, bmi=bmi, smoker=smoker)

    with col_result:
        st.subheader("💡 Estimated Expenditure")

        # Primary Prediction Display Card
        formatted_charge = f"${prediction:,.2f}"
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">Estimated Annual Medical Charge</div>
                <div class="prediction-value">{formatted_charge}</div>
                <div style="font-size: 0.9rem; opacity: 0.85;">Based on fitted OLS Linear Regression Pipeline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Risk Classification & Benchmark Indicators
        diff_from_median = prediction - BENCHMARK_MEDIAN_CHARGE
        median_comp = f"+${diff_from_median:,.2f}" if diff_from_median >= 0 else f"-${abs(diff_from_median):,.2f}"

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(
                label="Benchmark vs Dataset Median",
                value=f"${BENCHMARK_MEDIAN_CHARGE:,.2f}",
                delta=median_comp,
                delta_color="inverse",
            )
        with metric_col2:
            st.metric(
                label="Benchmark vs Dataset Mean",
                value=f"${BENCHMARK_MEAN_CHARGE:,.2f}",
                delta=f"{((prediction - BENCHMARK_MEAN_CHARGE) / BENCHMARK_MEAN_CHARGE) * 100:+.1f}%",
                delta_color="inverse",
            )

        # Risk Profile Badge
        st.markdown(
            f"""
            <div style="padding: 12px 16px; border-radius: 8px; background-color: #F7FAFC; border: 1px solid #E2E8F0; margin-top: 10px;">
                <span style="font-weight: 700; color: #2D3748;">Actuarial Risk Tier:</span>
                <span style="font-weight: 700; color: {risk_info['risk_color']}; margin-left: 6px;">{risk_info['risk_level']}</span>
                <p style="font-size: 0.88rem; color: #4A5568; margin-top: 6px; margin-bottom: 0;">{risk_info['risk_summary']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Tabular Input Verification
        with st.expander("🔍 View Raw Input Vector", expanded=False):
            summary_html = f"""
            <table class="profile-summary-table">
                <tr><td class="label">Age</td><td>{age} years</td></tr>
                <tr><td class="label">Biological Sex</td><td>{sex.capitalize()}</td></tr>
                <tr><td class="label">BMI</td><td>{bmi:.1f} kg/m² ({risk_info['bmi_category']})</td></tr>
                <tr><td class="label">Covered Children</td><td>{children}</td></tr>
                <tr><td class="label">Smoking Status</td><td>{'Yes (Smoker)' if smoker == 'yes' else 'No (Non-Smoker)'}</td></tr>
                <tr><td class="label">US Geographic Region</td><td>{region.capitalize()}</td></tr>
            </table>
            """
            st.markdown(summary_html, unsafe_allow_html=True)

    # 7. Mandatory Legal and Educational Disclaimer
    st.markdown(
        f"""
        <div class="disclaimer-container">
            <strong>Legal Notice:</strong> {MANDATORY_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    if st is not None:
        main()
    else:
        print("Notice: Streamlit package is required to run the interactive UI.", file=sys.stderr)
        print("Run 'streamlit run app.py' in an environment with Streamlit installed.", file=sys.stderr)
        # CLI fallback test
        try:
            m = load_model()
            sample_pred = predict_insurance_cost(
                model=m,
                age=30,
                sex="female",
                bmi=28.0,
                children=0,
                smoker="no",
                region="northeast",
            )
            print(f"Fallback CLI verification: Estimated Charge for default profile = ${sample_pred:,.2f}")
            print(f"Mandatory Disclaimer: {MANDATORY_DISCLAIMER}")
        except Exception as e:
            print(f"Model test error: {e}", file=sys.stderr)
            sys.exit(1)
