import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Hotel Cancellation Predictor", initial_sidebar_state="expanded")

MEAL_MAP = {
    "Not Selected": "Not Selected",
    "Meal Plan 1 (Breakfast Only)": "Meal Plan 1",
    "Meal Plan 2 (Lunch Included)": "Meal Plan 2",
    "Meal Plan 3 (Dinner Included)": "Meal Plan 3",
}

ROOM_MAP = {
    "Simple": "Room_Type 1",
    "Standard": "Room_Type 2",
    "Comfort": "Room_Type 3",
    "Superior": "Room_Type 4",
    "Deluxe": "Room_Type 5",
    "VIP 1": "Room_Type 6",
    "VIP 2": "Room_Type 7",
}

MEAL_OPTIONS = list(MEAL_MAP.keys())
ROOM_OPTIONS = list(ROOM_MAP.keys())
SEGMENT_OPTIONS = ["Online", "Offline", "Corporate", "Aviation", "Complementary"]

MONTHS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

CANCEL_THRESHOLD = 0.40
EUR_TO_USD = 1.10

CAT_COLS = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"]

NUM_COLS = [
    "no_of_adults", "no_of_children",
    "no_of_weekend_nights", "no_of_week_nights",
    "required_car_parking_space", "lead_time",
    "arrival_month", "arrival_date",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
    "total_nights",
    "total_guests",
    "cancellation_rate",
]


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("Hotel Reservations.csv")
    df["target"] = (df["booking_status"] == "Canceled").astype(int)

    df["total_nights"] = df["no_of_weekend_nights"] + df["no_of_week_nights"]
    df["total_guests"] = df["no_of_adults"] + df["no_of_children"]

    total_prior = (
        df["no_of_previous_cancellations"]
        + df["no_of_previous_bookings_not_canceled"]
    )

    df["cancellation_rate"] = (
        df["no_of_previous_cancellations"] / total_prior.replace(0, np.nan)
    ).fillna(0)

    return df


@st.cache_resource(show_spinner=False)
def train_model():
    df = load_data()

    X = df[CAT_COLS + NUM_COLS]
    y = df["target"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ))
    ])

    model.fit(X_train, y_train)
    return model


with st.spinner("Loading model..."):
    model = train_model()

with st.sidebar:
    st.header("Booking Details")

    no_adults = st.number_input("Adults staying", 1, 10, 2)
    no_children = st.number_input("Children staying", 0, 10, 0)
    repeated_guest = st.selectbox("Returning guest", ["No", "Yes"])
    prev_cancels = st.number_input("Previous cancellations", 0, 13, 0)
    prev_ok = st.number_input("Successful previous bookings", 0, 58, 0)
    special_req = st.slider("Special requests count", 0, 5, 0)

    weekend_nights = st.number_input("Weekend nights booked", 0, 7, 1)
    week_nights = st.number_input("Weekday nights booked", 0, 17, 2)
    meal_display = st.selectbox("Meal plan selected", MEAL_OPTIONS)
    room_display = st.selectbox("Room type booked", ROOM_OPTIONS)
    car_parking = st.selectbox("Car parking required", ["No", "Yes"])
    avg_price_usd = st.number_input("Average room price per night (USD)", 0.0, 660.0, 110.0, step=5.0)

    lead_time = st.number_input("Days before arrival booking was made", 0, 500, 30)
    booking_type = st.selectbox("Booking type", SEGMENT_OPTIONS)
    arrival_month = st.selectbox(
        "Arrival month",
        list(MONTHS.keys()),
        format_func=lambda x: MONTHS[x],
        index=5
    )
    arrival_date = st.number_input("Arrival day of month", 1, 31, 15)

    predict_button = st.button("Generate Prediction")

st.title("Hotel Cancellation Predictor")
st.write("This app gives a simple estimate of whether a booking is likely to be cancelled.")

if predict_button:
    total_nights = int(weekend_nights) + int(week_nights)
    total_guests = int(no_adults) + int(no_children)
    total_prior = int(prev_cancels) + int(prev_ok)

    if total_prior == 0:
        cancellation_rate = 0
    else:
        cancellation_rate = int(prev_cancels) / total_prior

    if total_guests < 1:
        st.warning("Booking should include at least one guest.")

    if total_nights < 1:
        st.warning("Please enter at least one night.")

    meal_value = MEAL_MAP[meal_display]
    room_value = ROOM_MAP[room_display]
    repeated_guest_value = 1 if repeated_guest == "Yes" else 0
    parking_value = 1 if car_parking == "Yes" else 0
    avg_price_eur = avg_price_usd / EUR_TO_USD

    input_df = pd.DataFrame([{
        "type_of_meal_plan": meal_value,
        "room_type_reserved": room_value,
        "market_segment_type": booking_type,
        "no_of_adults": float(no_adults),
        "no_of_children": float(no_children),
        "no_of_weekend_nights": float(weekend_nights),
        "no_of_week_nights": float(week_nights),
        "required_car_parking_space": float(parking_value),
        "lead_time": float(lead_time),
        "arrival_month": float(arrival_month),
        "arrival_date": float(arrival_date),
        "repeated_guest": float(repeated_guest_value),
        "no_of_previous_cancellations": float(prev_cancels),
        "no_of_previous_bookings_not_canceled": float(prev_ok),
        "avg_price_per_room": float(avg_price_eur),
        "no_of_special_requests": float(special_req),
        "total_nights": float(total_nights),
        "total_guests": float(total_guests),
        "cancellation_rate": float(cancellation_rate),
    }])

    proba = model.predict_proba(input_df)[0]
    cancel_prob = round(proba[1] * 100, 1)
    proceed_prob = round(proba[0] * 100, 1)

    if proba[1] >= CANCEL_THRESHOLD:
        st.error(f"Likely to cancel ({cancel_prob}%)")
    else:
        st.success(f"Likely to continue ({proceed_prob}%)")

    st.write(f"Cancellation probability: **{cancel_prob}%**")
    st.write(f"Proceed probability: **{proceed_prob}%**")

    st.write("This result is based on patterns the model found in earlier booking data.")

else:
    st.info("Enter the booking details in the sidebar and click Generate Prediction.")

with st.expander("How this works"):
    st.write(
        "The app trains a Random Forest model on past hotel reservation data and uses it to estimate "
        "whether a booking is likely to be cancelled."
    )


