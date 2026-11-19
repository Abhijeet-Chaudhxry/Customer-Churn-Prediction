from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# CHURNWISE | Customer Retention Intelligence Platform
# ------------------------------------------------------------

st.set_page_config(
    page_title="CHURNWISE",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churnwise_model.pkl"

# Keep this list aligned with the columns used to train the model.
FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
YES_LABEL = "Yes"

# ----------------------------- Styling -----------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #070A10;
        --panel: #0D1420;
        --panel2: #101B2B;
        --line: #1C2B40;
        --blue: #3998FF;
        --blue2: #77C2FF;
        --text: #F1F5FB;
        --muted: #91A1B6;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] { background: rgba(7,10,16,0.92); }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container {
        max-width: 1180px;
        padding-top: 2.9rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, .brand {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.035em;
        color: var(--text);
    }

    p, label, span, div { }
    [data-testid="stMarkdownContainer"] p { color: #B3C0D0; }

    .brand-wrap {
        display:flex; align-items:center; gap:10px;
        margin: 0 0 2.2rem 0;
    }
    .brand-mark {
        width:34px; height:34px; border-radius:10px;
        display:flex; align-items:center; justify-content:center;
        color:#8BCBFF; font-size:20px; font-weight:700;
        background:linear-gradient(145deg,#102A4B,#0C1727);
        border:1px solid #24558A;
        box-shadow:0 0 18px rgba(40,139,255,.14);
    }
    .brand-name {
        font-family:'Space Grotesk',sans-serif;
        font-size:17px; font-weight:700; letter-spacing:.13em;
        color:#F3F7FC;
    }
    .eyebrow {
        color:#70B9FF; text-transform:uppercase; letter-spacing:.16em;
        font-size:11px; font-weight:700; margin-bottom:10px;
    }
    .hero-title {
        font-family:'Space Grotesk',sans-serif;
        font-size:clamp(30px,4vw,48px); line-height:1.08;
        letter-spacing:-.055em; color:#F5F8FD; font-weight:700;
        margin:0 0 12px 0;
    }
    .hero-copy { color:#91A1B6; font-size:15px; max-width:650px; line-height:1.65; }
    .section-label {
        color:#7DBEFF; font-size:11px; font-weight:700;
        letter-spacing:.15em; text-transform:uppercase;
        margin: 8px 0 12px 0;
    }
    .choice-card {
        background:linear-gradient(145deg,#101B2B,#0B111B);
        border:1px solid #21334B; border-radius:16px;
        padding:19px 20px; min-height:138px;
        transition:all .2s ease;
    }
    .choice-card:hover {
        border-color:#398FE8;
        box-shadow:0 0 24px rgba(38,132,238,.12);
        transform:translateY(-1px);
    }
    /* The landing choices are real Streamlit buttons, not decorative HTML cards. */
    .st-key-single_choice button, .st-key-multiple_choice button {
        width:100%; min-height:148px; height:auto; white-space:pre-wrap;
        text-align:left; justify-content:flex-start; align-items:flex-start;
        padding:22px 22px; border-radius:16px;
        background:linear-gradient(145deg,#101B2B,#0B111B);
        border:1px solid #21334B; box-shadow:none;
        color:#F0F5FC; line-height:1.65;
    }
    .st-key-single_choice button:hover, .st-key-multiple_choice button:hover {
        background:linear-gradient(145deg,#13243A,#0B111B);
        border-color:#398FE8; color:#F0F5FC;
        box-shadow:0 0 24px rgba(38,132,238,.12);
    }
    .choice-icon { color:#72BDFF; font-size:23px; margin-bottom:12px; }
    .choice-title { color:#F0F5FC; font-size:17px; font-weight:700; margin-bottom:5px; }
    .choice-desc { color:#91A1B6; font-size:12px; line-height:1.5; }
    .subtle-panel {
        background:#0C1420; border:1px solid #1B2B40;
        border-radius:14px; padding:17px 19px;
    }
    .step-pill {
        display:inline-block; border:1px solid #214B77; color:#82C4FF;
        background:#0C1C30; padding:5px 10px; border-radius:30px;
        font-size:11px; font-weight:700; letter-spacing:.08em;
    }
    .result-card {
        background:linear-gradient(145deg,#101C2C,#0B121D);
        border:1px solid #20334B; border-radius:15px; padding:18px;
    }
    div.stButton > button, div.stFormSubmitButton > button {
        background:linear-gradient(100deg,#1769BD,#258FEF);
        color:white; border:1px solid #3C9EFF; border-radius:10px;
        min-height:44px; font-weight:700; transition:all .2s ease;
        box-shadow:0 0 18px rgba(31,132,239,.13);
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        border-color:#91D0FF; color:white;
        box-shadow:0 0 22px rgba(48,151,255,.28);
    }
    div[data-testid="stFileUploader"] {
        background:#0C1420; border:1px dashed #2A527A;
        border-radius:13px; padding:12px;
    }
    [data-testid="stMetric"] {
        background:#0D1725; border:1px solid #1D3048;
        padding:15px 16px; border-radius:13px;
    }
    [data-testid="stMetricLabel"] { color:#9AABC0; }
    [data-testid="stMetricValue"] { color:#F2F7FD; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background:#0C1420 !important; border-color:#263A53 !important;
    }
    input, textarea { color:#F3F7FC !important; }
    [data-testid="stTabs"] button { color:#AAB8C9; }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color:#79BEFF; border-bottom-color:#3998FF;
    }
    hr { border-color:#1A293B; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Helpers -----------------------------

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def normalize_customer_frame(frame: pd.DataFrame):
    """Validate and align an uploaded dataset to the model's training schema."""
    df = frame.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Dataset exports may include these non-feature columns.
    for extra in ["customerID", "Churn"]:
        if extra in df.columns:
            df = df.drop(columns=extra)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return None, missing

    df = df[FEATURES].copy()

    # Match the original training cleanup for blank TotalCharges.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True),
        errors="coerce",
    )
    # Fix blank TotalCharges for customers with zero tenure
    mask = df["tenure"].eq(0) & df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = 0
    
    for col in ["tenure", "MonthlyCharges"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_numeric = [c for c in NUMERIC_COLUMNS if df[c].isna().any()]
    if invalid_numeric:
        return None, [f"{c} contains blank or non-numeric values" for c in invalid_numeric]

    # SeniorCitizen is a categorical model input despite being stored as 0/1.
    df["SeniorCitizen"] = df["SeniorCitizen"].replace(
        {"0": 0, "1": 1, "0.0": 0, "1.0": 1}
    )
    return df, []


def read_uploaded_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    raise ValueError("Please upload a CSV or Excel file.")


def predict_frame(model, threshold, frame):
    probabilities = model.predict_proba(frame)
    yes_index = list(model.classes_).index(YES_LABEL)
    churn_probability = probabilities[:, yes_index]
    labels = np.where(churn_probability >= threshold, "Yes", "No")
    result = frame.copy()
    result.insert(0, "Churn Probability", churn_probability)
    result.insert(1, "Predicted Churn", labels)
    result.insert(2, "Risk Level", np.where(churn_probability >= threshold, "At risk", "Lower risk"))
    return result


def retention_note(row):
    """Simple rule-based prompts, not causal explanations or guaranteed remedies."""
    notes = []
    if row.get("Contract") == "Month-to-month":
        notes.append("Review whether a suitable longer-term plan would benefit this customer.")
    if row.get("tenure", 0) <= 12:
        notes.append("Consider an early-tenure check-in to understand the customer's experience.")
    if row.get("TechSupport") == "No" and row.get("InternetService") != "No":
        notes.append("Check whether the customer is aware of available technical support.")
    if row.get("OnlineSecurity") == "No" and row.get("InternetService") != "No":
        notes.append("Consider whether an online-security option is relevant to the customer's needs.")
    if not notes:
        notes.append("Review recent service interactions and customer feedback before choosing an intervention.")
    return notes[:3]


def show_prediction(row, probability, threshold):
    is_risk = probability >= threshold
    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("Estimated churn probability", f"{probability:.1%}")
    with c2:
        st.metric("Model classification", "At risk" if is_risk else "Lower risk")
    st.progress(float(np.clip(probability, 0, 1)))
    if is_risk:
        st.warning(
            f"The estimated probability is above the {threshold:.0%} operating threshold. "
            "This is a model flag for review, not a certainty that the customer will leave."
        )
    else:
        st.info(
            f"The estimated probability is below the {threshold:.0%} operating threshold. "
            "This does not guarantee that the customer will stay."
        )
    st.markdown("#### Suggested retention checks")
    for item in retention_note(row):
        st.write(f"• {item}")


def header(step=None):
    if step is None:
        st.markdown(
            """
            <div class="brand-wrap">
                <div class="brand-mark">◈</div>
                <div class="brand-name">CHURNWISE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f'<span class="step-pill">{step}</span>', unsafe_allow_html=True)


def make_manual_form():
    with st.form("manual_customer_form"):
        st.markdown("#### Customer profile")
        a, b, c = st.columns(3)
        with a:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Has partner", ["Yes", "No"])
            dependents = st.selectbox("Has dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        with b:
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
            tech = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
        with c:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly = st.number_input("Monthly charges", min_value=0.0, value=70.0, step=1.0)
            total_default = float(monthly) * int(tenure)
            total = st.number_input("Total charges", min_value=0.0, value=total_default, step=10.0)

        submitted = st.form_submit_button("Analyse customer →", use_container_width=True)

    if submitted:
        return pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }])[FEATURES]
    return None


# The saved model is loaded only when a prediction is requested.

# ----------------------------- Landing / flow -----------------------------

header()
if "input_kind" not in st.session_state:
    st.session_state.input_kind = None
if "input_df" not in st.session_state:
    st.session_state.input_df = None
if "input_source" not in st.session_state:
    st.session_state.input_source = None
if "active_module" not in st.session_state:
    st.session_state.active_module = None

# Use Streamlit callbacks for navigation. This avoids relying on button-return
# branches followed by an explicit rerun, and keeps state changes atomic.
def choose_single_customer():
    st.session_state.input_kind = "single"
    st.session_state.input_df = None
    st.session_state.input_source = None
    st.session_state.pop("module_choice", None)


def choose_multiple_customers():
    st.session_state.input_kind = "multiple"
    st.session_state.input_df = None
    st.session_state.input_source = None
    st.session_state.pop("module_choice", None)


def reset_analysis():
    st.session_state.input_kind = None
    st.session_state.input_df = None
    st.session_state.input_source = None
    st.session_state.active_module = None
    st.session_state.pop("module_choice", None)


if st.session_state.input_kind is None:
    st.markdown('<div class="eyebrow">Customer retention intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">What would you like to analyse?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">Choose an option to get started.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown('<div class="section-label">01 / Choose your data</div>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.button(
            "◉  Single Customer Analysis →",
            key="single_choice", use_container_width=True,
            on_click=choose_single_customer,
        )
    with right:
        st.button(
            "◎  Multiple Customer Analysis →",
            key="multiple_choice", use_container_width=True,
            on_click=choose_multiple_customers,
        )

    st.markdown("")
    st.caption("Built for better decisions · Created by Abhijeet Chaudhary")

elif st.session_state.input_df is None:
    kind = st.session_state.input_kind
    header("STEP 02  /  PROVIDE CUSTOMER DATA")
    st.markdown("### " + ("Add one customer" if kind == "single" else "Upload your customer dataset"))
    st.caption("We'll take care of preparing it for analysis.")

    st.button("← Back to start", key="back_to_start", on_click=reset_analysis)

    if kind == "single":
        entry_method = st.radio(
            "How would you like to provide the customer details?",
            ["Enter manually", "Upload a file"],
            horizontal=True,
        )
        if entry_method == "Enter manually":
            manual_df = make_manual_form()
            if manual_df is not None:
                st.session_state.input_df = manual_df
                st.session_state.input_source = "single"
                st.rerun()
        else:
            uploaded = st.file_uploader("Upload a CSV or Excel file with exactly one customer", type=["csv", "xlsx", "xls"])
            if uploaded is not None and st.button("Validate and continue →", use_container_width=True):
                try:
                    raw = read_uploaded_file(uploaded)
                    clean, missing = normalize_customer_frame(raw)
                    if missing:
                        st.error("Missing required columns: " + ", ".join(missing))
                    elif len(clean) != 1:
                        st.error(f"This option expects exactly one customer. Your file contains {len(clean)} rows.")
                    else:
                        st.session_state.input_df = clean
                        st.session_state.input_source = "single"
                        st.rerun()
                except Exception as exc:
                    st.error(f"Could not read this file: {exc}")
    else:
        uploaded = st.file_uploader("Upload customer dataset", type=["csv", "xlsx", "xls"])
        if uploaded is not None and st.button("Validate dataset →", use_container_width=True):
            try:
                raw = read_uploaded_file(uploaded)
                clean, missing = normalize_customer_frame(raw)
                if missing:
                    st.error("Missing required columns: " + ", ".join(missing))
                elif len(clean) < 2:
                    st.error("This option needs at least two customer rows. Choose Single Customer for one row.")
                else:
                    st.session_state.input_df = clean
                    st.session_state.input_source = "multiple"
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not read this file: {exc}")

# ----------------------------- Results and modules -----------------------------

if st.session_state.input_df is not None:
    customer_df = st.session_state.input_df
    source = st.session_state.input_source or st.session_state.input_kind

    try:
        artifact = load_artifact()
        model = artifact["model"]
        threshold = float(artifact.get("threshold", 0.35))
        model_features = artifact.get("features", FEATURES)
        if list(model_features) != FEATURES:
            FEATURES[:] = list(model_features)
        predictions = predict_frame(model, threshold, customer_df)
    except FileNotFoundError:
        st.error(f"The customer data is ready, but `{MODEL_PATH.name}` is missing. Put the trained model file in the same folder as app.py, then restart CHURNWISE.")
        st.stop()
    except Exception as exc:
        st.error("Prediction failed. Check that the uploaded data uses the same categories and structure as the training dataset.")
        st.code(str(exc))
        st.stop()

    st.divider()
    header("STEP 03  /  EXPLORE RESULTS")
    st.markdown("### Your analysis is ready")
    st.caption(f"{len(customer_df):,} customer record(s) processed successfully.")

    modules = ["Individual Prediction"]
    if source == "multiple" and len(customer_df) > 1:
        modules.append("Customer Analytics")

    selected = st.radio(
        "Choose what you want to explore",
        modules,
        horizontal=True,
        key="module_choice",
    )

    if selected == "Individual Prediction":
        if len(customer_df) == 1:
            selected_index = 0
        else:
            selected_index = st.selectbox(
                "Select a customer record",
                options=list(range(len(customer_df))),
                format_func=lambda i: f"Customer {i + 1}",
            )

        row = customer_df.iloc[selected_index]
        result_row = predictions.iloc[selected_index]
        st.markdown('<div class="section-label">INDIVIDUAL CUSTOMER RESULT</div>', unsafe_allow_html=True)
        show_prediction(row, float(result_row["Churn Probability"]), threshold)

        with st.expander("View customer details"):
            st.dataframe(row.to_frame("Value"), use_container_width=True)

    elif selected == "Customer Analytics":
        churn_count = int((predictions["Predicted Churn"] == "Yes").sum())
        lower_count = len(predictions) - churn_count
        avg_probability = float(predictions["Churn Probability"].mean())

        st.markdown('<div class="section-label">CUSTOMER DATASET OVERVIEW</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Customers analysed", f"{len(predictions):,}")
        m2.metric("Flagged at risk", f"{churn_count:,}")
        m3.metric("Lower risk", f"{lower_count:,}")
        m4.metric("Average estimated risk", f"{avg_probability:.1%}")

        chart_left, chart_right = st.columns(2)
        with chart_left:
            risk_counts = predictions["Predicted Churn"].value_counts().rename_axis("Prediction").reset_index(name="Customers")
            fig = px.pie(
                risk_counts, names="Prediction", values="Customers", hole=0.68,
                color="Prediction",
                color_discrete_map={"Yes": "#3998FF", "No": "#25364D"},
            )
            fig.update_layout(
                title="Predicted customer groups",
                paper_bgcolor="#070A10", plot_bgcolor="#070A10",
                font_color="#DCE7F4", legend_title_text="",
                margin=dict(l=10, r=10, t=55, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        with chart_right:
            fig2 = px.histogram(
                predictions, x="Churn Probability", nbins=20,
                title="Distribution of estimated churn probabilities",
                color_discrete_sequence=["#3998FF"],
            )
            fig2.add_vline(x=threshold, line_dash="dash", line_color="#8CCBFF",
                           annotation_text=f"Threshold {threshold:.0%}")
            fig2.update_layout(
                paper_bgcolor="#070A10", plot_bgcolor="#0D1420",
                font_color="#DCE7F4",
                xaxis=dict(gridcolor="#1B2B40", tickformat=".0%"),
                yaxis=dict(gridcolor="#1B2B40"),
                margin=dict(l=10, r=10, t=55, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Customer-level predictions")
        display_df = predictions.copy()
        display_df["Churn Probability"] = display_df["Churn Probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv_bytes = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download prediction results (CSV)",
            data=csv_bytes,
            file_name="churnwise_predictions.csv",
            mime="text/csv",
        )

    st.divider()
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.caption("Predictions are estimates from the trained model, not guarantees of future customer behaviour.")
    with col_b:
        st.button("Start a new analysis", use_container_width=True, on_click=reset_analysis)
