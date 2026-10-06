import joblib
import pandas as pd
import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="ED Waiting Predictor", layout="centered")
st.title("ED Waiting Volume Predictor")
st.markdown("Predict the average number of persons waiting for emergency treatment based on environmental and epidemiological factors.")

# --- Cache Model Loading ---
@st.cache_resource
def load_model():
    return joblib.load(r"best_xgboost_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'best_xgboost_model.pkl' is in the same directory.")
    st.stop()

# --- Data Dictionaries ---
LHD_HOSPITAL_DATA = {
    "Central Coast": {"Gosford Hospital": 2250, "Wyong Hospital": 2259},
    "Far West": {"Broken Hill Health Service": 2880},
    "Hunter New England": {
        "John Hunter Hospital": 2305, "Maitland Hospital": 2320, 
        "Calvary Mater Newcastle": 2298, "Tamworth Hospital": 2340, 
        "Armidale Hospital": 2350, "Manning Base Hospital": 2430, 
        "Cessnock District Hospital": 2325, "Singleton District Hospital": 2330, 
        "Muswellbrook District Hospital": 2333
    },
    "Illawarra Shoalhaven": {
        "Wollongong Hospital": 2500, "Shellharbour Hospital": 2529, 
        "Shoalhaven District Memorial Hospital": 2541, "Bulli Hospital": 2516, 
        "Milton Ulladulla Hospital": 2539
    },
    "Mid North Coast": {
        "Coffs Harbour Health Campus": 2450, "Port Macquarie Base Hospital": 2444, 
        "Kempsey District Hospital": 2440, "Macksville Centre": 2447
    },
    "Murrumbidgee": {
        "Wagga Wagga Base Hospital": 2650, "Griffith Base Hospital": 2680, 
        "Deniliquin Health Service": 2710, "Tumut Health Service": 2720, 
        "Young Health Service": 2594
    },
    "Nepean Blue Mountains": {
        "Nepean Hospital": 2747, "Blue Mountains District ANZAC Memorial Hospital": 2780, 
        "Hawkesbury District Health Service": 2756, "Lithgow Hospital": 2790, 
        "Springwood Hospital": 2777
    },
    "Northern NSW": {
        "Tweed Valley Hospital": 2487, "Lismore Base Hospital": 2480, 
        "Grafton Base Hospital": 2460, "Ballina District Hospital": 2478, 
        "Murwillumbah District Hospital": 2484, "Casino & District Memorial Hospital": 2470, 
        "Maclean District Hospital": 2463
    },
    "Northern Sydney": {
        "Royal North Shore Hospital": 2065, "Hornsby Ku-ring-gai Hospital": 2077, 
        "Northern Beaches Hospital": 2086, "Ryde Hospital": 2112, "Mona Vale Hospital": 2103
    },
    "South Eastern Sydney": {
        "Prince of Wales Hospital": 2031, "St George Hospital": 2217, 
        "Sutherland Hospital": 2229, "Sydney / Sydney Eye Hospital": 2000
    },
    "South Western Sydney": {
        "Liverpool Hospital": 2170, "Campbelltown Hospital": 2560, 
        "Bankstown-Lidcombe Hospital": 2200, "Fairfield Hospital": 2176, 
        "Bowral & District Hospital": 2576, "Camden Hospital": 2570
    },
    "Southern NSW": {
        "South East Regional Hospital": 2550, "Goulburn Base Hospital": 2580, 
        "Queanbeyan Health Service": 2620, "Cooma Health Service": 2630, 
        "Eurobodalla Health Service": 2537
    },
    "Sydney": {
        "Royal Prince Alfred Hospital": 2050, "Concord Repatriation General Hospital": 2139, 
        "Canterbury Hospital": 2193, "Balmain Hospital": 2041
    },
    "Western NSW": {
        "Dubbo Base Hospital": 2830, "Orange Health Service": 2800, 
        "Bathurst Base Hospital": 2795, "Mudgee Health Service": 2850, 
        "Parkes Health Service": 2870, "Forbes Health Service": 2871, 
        "Cowra Health Service": 2794
    },
    "Western Sydney": {
        "Westmead Hospital": 2145, "Blacktown Hospital": 2148, 
        "Mount Druitt Hospital": 2770, "Auburn Hospital": 2144
    },
}

# --- Inputs: Facility & Timing ---
with st.container(border=True):
    st.subheader("Facility & Timing")
    col_lhd, col_hosp = st.columns(2)

    with col_lhd:
        selected_lhd = st.selectbox("Local Health District (LHD)", sorted(list(LHD_HOSPITAL_DATA.keys())))

    hospital_options = sorted(list(LHD_HOSPITAL_DATA[selected_lhd].keys()))
    with col_hosp:
        selected_hospital = st.selectbox("Hospital Name", hospital_options)

    selected_postcode = LHD_HOSPITAL_DATA[selected_lhd][selected_hospital]

    day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

# --- Inputs: Epidemiological Surveillance ---
with st.container(border=True):
    st.subheader("Infectious Disease Surveillance")
    combined_cases = st.number_input(
        "Combined Infectious Cases (COVID + Influenza + RSV)",
        min_value=0, value=20, step=1,
    )

    covid_cases = int(combined_cases * 0.4)
    rsv_cases = int(combined_cases * 0.3)
    flu_cases = int(combined_cases * 0.3)

# --- Inputs: Meteorological Conditions ---
with st.container(border=True):
    st.subheader("Weather Conditions")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        temp_min = st.slider("Min Temperature (°C)", -5.0, 35.0, 15.0, 0.5)
    with col_t2:
        temp_max = st.slider("Max Temperature (°C)", 0.0, 48.0, 24.0, 0.5)

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        rain = st.number_input("Precipitation (mm)", min_value=0.0, max_value=200.0, value=0.0, step=0.5)
    with col_w2:
        humidity_mean = st.slider("Mean Relative Humidity (%)", 10, 100, 65)

    humidity_min = max(humidity_mean - 15, 10)
    humidity_max = min(humidity_mean + 15, 100)

# --- Live Prediction ---
st.divider()

input_data = {
    "Postcode": [selected_postcode],
    "COVID Cases": [covid_cases],
    "RSV Cases": [rsv_cases],
    "Influenza Cases": [flu_cases],
    "Combined Cases": [combined_cases],
    "temperature_max": [temp_max],
    "temperature_min": [temp_min],
    "precipitation_sum": [rain],
    "relative_humidity_min": [humidity_min],
    "relative_humidity_max": [humidity_max],
    "relative_humidity_mean": [humidity_mean],
    "Day of Week": [day],
    "LHD": [selected_lhd],
    "Hospital Name": [selected_hospital],
}
input_df = pd.DataFrame(input_data)

X_encoded = pd.get_dummies(input_df, columns=["Day of Week", "LHD", "Hospital Name"], drop_first=False)

bool_cols = X_encoded.select_dtypes(include=["bool"]).columns
X_encoded[bool_cols] = X_encoded[bool_cols].astype(int)

try:
    model_features = model.get_booster().feature_names
except AttributeError:
    model_features = list(model.feature_names_in_)

X_encoded = X_encoded.reindex(columns=model_features, fill_value=0)

pred = model.predict(X_encoded)[0]

col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
with col_res2:
    st.metric(
        label=f"Estimated Average Persons Waiting at {selected_hospital}",
        value=f"{max(0.0, float(pred)):.1f}"
    )