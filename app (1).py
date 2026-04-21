import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

#  page config 

st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    layout="wide"
)

# feature lists 

CAT_FEATURES = [
    'type_of_meal_plan',
    'room_type_reserved',
    'market_segment_type'
]

NUM_FEATURES = [
    'no_of_adults',
    'no_of_children',
    'no_of_weekend_nights',
    'no_of_week_nights',
    'required_car_parking_space',
    'lead_time',
    'arrival_month',
    'arrival_date',
    'repeated_guest',
    'no_of_previous_cancellations',
    'no_of_previous_bookings_not_canceled',
    'avg_price_per_room',
    'no_of_special_requests',
    'total_nights',
    'total_guests',
    'cancellation_rate'
]

CANCEL_THRESHOLD = 0.40

#load data and train model 

@st.cache_resource(show_spinner=False)
def train_model():
    # load and prepare dataset
    df = pd.read_csv('Hotel_Reservations.csv')

    # create target: cancelled = 1
    df['target'] = (df['booking_status'] == 'Canceled').astype(int)

    # feature engineering
    df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
    df['total_guests']  = df['no_of_adults'] + df['no_of_children']
    total_prev = df['no_of_previous_cancellations'] + df['no_of_previous_bookings_not_canceled']
    df['cancellation_rate'] = (df['no_of_previous_cancellations'] / total_prev).fillna(0)

    # select features - dropping arrival_year (only 2017/2018, not useful)
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df['target']

    # stratified split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # preprocessing
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, NUM_FEATURES),
        ('cat', cat_pipeline, CAT_FEATURES)
    ])

    # build and train the model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    # calculate train and test accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))

    return model, round(train_acc, 4), round(test_acc, 4)


# load model 

with st.spinner("Loading model... please wait"):
    model, train_acc, test_acc = train_model()

#  sidebar

st.sidebar.title("Booking Details")
st.sidebar.markdown("Enter the booking information below.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Guest Details**")

# taking input from user
no_adults  = st.sidebar.number_input("Adults staying", min_value=1, max_value=10, value=2)
no_children = st.sidebar.number_input("Children staying", min_value=0, max_value=10, value=0)
repeated   = st.sidebar.selectbox("Returning guest", ["No", "Yes"])
prev_cancel = st.sidebar.number_input("Previous cancellations", min_value=0, max_value=13, value=0)
prev_ok    = st.sidebar.number_input("Successful previous bookings", min_value=0, max_value=58, value=0)
special_req = st.sidebar.slider("Special requests", min_value=0, max_value=5, value=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Stay Details**")

weekend_nights = st.sidebar.number_input("Weekend nights", min_value=0, max_value=7, value=1)
week_nights    = st.sidebar.number_input("Weekday nights", min_value=0, max_value=17, value=2)
meal_plan      = st.sidebar.selectbox("Meal plan", ["Not Selected", "Meal Plan 1", "Meal Plan 2", "Meal Plan 3"])
room_type      = st.sidebar.selectbox("Room type", ["Room_Type 1", "Room_Type 2", "Room_Type 3",
                                                     "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
parking        = st.sidebar.selectbox("Car parking required", ["No", "Yes"])
avg_price      = st.sidebar.number_input("Average room price per night (USD)", min_value=0.0, max_value=600.0,
                                          value=100.0, step=5.0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Booking and Arrival**")

lead_time   = st.sidebar.number_input("Days before arrival booking was made", min_value=0, max_value=500, value=30)
market_seg  = st.sidebar.selectbox("Booking type", ["Online", "Offline", "Corporate", "Aviation", "Complementary"])
arr_month   = st.sidebar.selectbox("Arrival month", list(range(1, 13)),
                                    format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                                           "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
arr_date    = st.sidebar.number_input("Arrival day of month", min_value=1, max_value=31, value=15)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("Predict", use_container_width=True)

# main area 
st.title("Hotel Cancellation Predictor")
st.markdown("Predicts whether a hotel reservation is likely to be cancelled based on booking details.")
st.markdown("---")

# show model accuracy at top
col_a, col_b, col_c = st.columns(3)
col_a.metric("Model", "Random Forest")
col_b.metric("Train Accuracy", f"{train_acc:.1%}")
col_c.metric("Test Accuracy",  f"{test_acc:.1%}")

st.markdown("---")

# prediction

if predict_btn:
    # converting all inputs into a dataframe for the model
    repeated_val = 1 if repeated == "Yes" else 0
    parking_val  = 1 if parking  == "Yes" else 0

    total_n = weekend_nights + week_nights
    total_g = no_adults + no_children

    total_prev_bookings = prev_cancel + prev_ok
    cancel_rate = prev_cancel / total_prev_bookings if total_prev_bookings > 0 else 0.0

    # price entered in USD - convert to EUR for model (dataset uses EUR)
    avg_price_eur = avg_price / 1.10

    input_data = pd.DataFrame([{
        'no_of_adults':                         float(no_adults),
        'no_of_children':                       float(no_children),
        'no_of_weekend_nights':                 float(weekend_nights),
        'no_of_week_nights':                    float(week_nights),
        'required_car_parking_space':           float(parking_val),
        'lead_time':                            float(lead_time),
        'arrival_month':                        float(arr_month),
        'arrival_date':                         float(arr_date),
        'repeated_guest':                       float(repeated_val),
        'no_of_previous_cancellations':         float(prev_cancel),
        'no_of_previous_bookings_not_canceled': float(prev_ok),
        'avg_price_per_room':                   float(avg_price_eur),
        'no_of_special_requests':               float(special_req),
        'total_nights':                         float(total_n),
        'total_guests':                         float(total_g),
        'cancellation_rate':                    float(cancel_rate),
        'type_of_meal_plan':                    meal_plan,
        'room_type_reserved':                   room_type,
        'market_segment_type':                  market_seg
    }])

    # get probabilities
    proba       = model.predict_proba(input_data)[0]
    cancel_p    = round(proba[1] * 100, 1)
    proceed_p   = round(proba[0] * 100, 1)
    is_cancelled = proba[1] >= CANCEL_THRESHOLD

    # show result
    col1, col2 = st.columns([3, 2])

    with col1:
        if is_cancelled:
            st.error(f"**Reservation Shows Elevated Cancellation Risk**")
            st.markdown(
                f"Based on the details provided, this booking has a **{cancel_p}%** estimated "
                f"probability of cancellation. Consider sending a confirmation request or "
                f"reviewing the booking before the arrival date."
            )
        else:
            st.success(f"**Booking Shows Low Risk of Cancellation**")
            st.markdown(
                f"Based on the details provided, this booking has a **{proceed_p}%** estimated "
                f"probability of proceeding to check-in."
            )

    with col2:
        st.markdown("**Confidence Breakdown**")
        st.markdown(f"Booking Proceeds: **{proceed_p}%**")
        st.progress(proceed_p / 100)
        st.markdown(f"Cancelled: **{cancel_p}%**")
        st.progress(cancel_p / 100)

    # summary table
    st.markdown("---")
    st.markdown("### Summary of Inputs Used")

    MONTH_NAMES = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                   7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}

    summary = pd.DataFrame({
        "Feature": [
            "Adults staying", "Children staying", "Weekend nights", "Weekday nights",
            "Meal plan", "Room type", "Car parking", "Lead time (days)",
            "Arrival month", "Arrival day", "Booking type", "Returning guest",
            "Previous cancellations", "Successful previous bookings",
            "Avg room price (USD)", "Special requests"
        ],
        "Value Entered": [
            no_adults, no_children, weekend_nights, week_nights,
            meal_plan, room_type, parking, lead_time,
            MONTH_NAMES[arr_month], arr_date, market_seg, repeated,
            prev_cancel, prev_ok,
            f"${avg_price:,.2f}", special_req
        ]
    })

    st.dataframe(summary, use_container_width=True, hide_index=True)

else:
    # placeholder before prediction is run
    st.info("Fill in the booking details in the sidebar and click **Predict** to get a result.")
