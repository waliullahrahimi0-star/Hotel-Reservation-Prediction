import streamlit as st
import pandas as pd
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Hotel Cancellation Predictor", initial_sidebar_state="expanded")


# Using the 40% threshold because missing a real cancellation costs the hotel more than a false alarm
CANCEL_THRESHOLD = 0.40
#Balance te EUR and USD
EUR_TO_USD = 1.10

# Mapping display 
meal_map = {
    "Not Selected": "Not Selected",
    "Meal Plan 1 (Breakfast Only)": "Meal Plan 1",
    "Meal Plan 2 (Lunch Included)": "Meal Plan 2",
    "Meal Plan 3 (Dinner Included)": "Meal Plan 3",
}
room_map = {
    "Simple": "Room_Type 1",
    "Standard": "Room_Type 2",
    "Comfort": "Room_Type 3",
    "Superior": "Room_Type 4",
    "Deluxe": "Room_Type 5",
    "VIP 1": "Room_Type 6",
    "VIP 2": "Room_Type 7",
}
MEAL_DISPLAY_OPTIONS = list(meal_map.keys())
ROOM_DISPLAY_OPTIONS = list(room_map.keys())
SEGMENT_OPTIONS = ["Online", "Offline", "Corporate", "Aviation", "Complementary"]
MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

CATEGORICAL_COLS = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"]
NUMERICAL_COLS = [
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
]

# labels for the importance chart
_NUM_DISPLAY = {
    "no_of_adults": "Adults in booking",
    "no_of_children": "Children in booking",
    "no_of_weekend_nights": "Weekend nights",
    "no_of_week_nights": "Weekday nights",
    "required_car_parking_space": "Car parking",
    "lead_time": "Lead time (days)",
    "arrival_month": "Arrival month",
    "arrival_date": "Arrival day",
    "repeated_guest": "Returning guest",
    "no_of_previous_cancellations": "Previous cancellations",
    "no_of_previous_bookings_not_canceled": "Prior successful bookings",
    "avg_price_per_room": "Room price per night",
    "no_of_special_requests": "Special requests",
    "total_nights": "Total nights",
    "total_guests": "Total guests",
}
_CAT_DISPLAY = {
    "type_of_meal_plan": "Meal plan type",
    "room_type_reserved": "Room type",
    "market_segment_type": "Booking channel",
}


@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    df = pd.read_csv("Hotel_Reservations.csv")
    df["target"] = (df["booking_status"] == "Canceled").astype(int)
    # These two engineered features ended up being more useful than I expected
    df["total_nights"] = df["no_of_weekend_nights"] + df["no_of_week_nights"]
    df["total_guests"] = df["no_of_adults"] + df["no_of_children"]
    return df


@st.cache_resource(show_spinner=False)
def train_model():
    df = load_and_prepare_data()
    X = df[CATEGORICAL_COLS + NUMERICAL_COLS]
    y = df["target"]

    #by using train_test_split here balance the cancellation ratio
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Median works better than mean so we chose median
    num_transformer = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])

    # filling unknown categories with a word "Unknown" 
    cat_transformer = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    col_transformer = ColumnTransformer([
        ("num", num_transformer, NUMERICAL_COLS),
        ("cat", cat_transformer, CATEGORICAL_COLS),
    ])

    pipe = Pipeline([
        ("preprocessor", col_transformer),
        ("classifier", RandomForestClassifier(
            n_estimators=150, max_depth=12,
            min_samples_split=10, min_samples_leaf=5,
            random_state=42, class_weight="balanced", n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


@st.cache_data(show_spinner=False)
def get_top_importances(_mdl, n=7):
    # Explodes each categorical into many binary columns
    rf_clf = _mdl.named_steps["classifier"]
    ohe_names = (
        _mdl.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["enc"]
        .get_feature_names_out(CATEGORICAL_COLS)
        .tolist()
    )
    all_names = NUMERICAL_COLS + ohe_names
    importances = rf_clf.feature_importances_

    agg = {}
    for feat, imp in zip(all_names, importances):
        if feat in NUMERICAL_COLS:
            key = _NUM_DISPLAY.get(feat, feat)
        else:
            key = next(
                (_CAT_DISPLAY[c] for c in CATEGORICAL_COLS if feat.startswith(c + "_")),
                feat,
            )
        agg[key] = agg.get(key, 0.0) + imp

    series = pd.Series(agg).sort_values(ascending=False)
    return series.head(n)


def build_explanation(lead_t, special_req, avg_usd, prev_cancel,
                      bk_type, ret_guest, total_n, cancel_prob):
    # Convert EUR to USD 
    avg_eur = avg_usd / EUR_TO_USD
    drivers = []

    # Lead time is consistently one of the strongest signals in the data
    if lead_t > 150:
        drivers.append("a very long lead time")
    elif lead_t > 60:
        drivers.append("a moderately long lead time")
    elif lead_t < 14:
        drivers.append("a short lead time and is close to arrival")

    if special_req == 0:
        drivers.append("no special requests, which often means lower commitment to the stay")
    elif special_req >= 2:
        drivers.append("multiple special requests, which usually means the guest is more invested")

    if avg_eur > 200:
        drivers.append("a high nightly room rate")
    elif avg_eur < 70:
        drivers.append("a low nightly rate, which tends to go with more flexible bookings")

    if bk_type == "Online":
        drivers.append("an online booking channel, which casued to have higher cancellation rates")
    elif bk_type in ["Corporate", "Offline"]:
        drivers.append(f"a {bk_type.lower()} booking channel, where guests tend to follow through")

    if prev_cancel >= 2:
        drivers.append("a history of multiple prior cancellations")
    elif prev_cancel == 1:
        drivers.append("one prior cancellation on record")

    if ret_guest == "Yes":
        drivers.append("returning guest status, which is generally a good sign for follow-through")

    if total_n == 0:
        drivers.append("a booking with no confirmed nights")

    if not drivers:
        if cancel_prob >= CANCEL_THRESHOLD:
            return ("Nothing specific stands out, but this mix of inputs tends to match "
                    "bookings that end up getting cancelled.")
        else:
            return ("Nothing specific stands out, but this mix of inputs generally matches "
                    "bookings that go ahead without issue.")

    if len(drivers) == 1:
        body_str = drivers[0]
    elif len(drivers) == 2:
        body_str = f"{drivers[0]} and {drivers[1]}"
    else:
        body_str = ", ".join(drivers[:-1]) + ", and " + drivers[-1]

    if cancel_prob >= CANCEL_THRESHOLD:
        return (f"The main factors here are {body_str}. "
                "Bookings with this kind of profile tend to get cancelled more often.")
    else:
        return (f"Working in favour here: {body_str}. "
                "Bookings like this tend to go ahead.")


def get_key_drivers(lead_t, special_req, avg_usd, prev_cancel,
                    bk_type, ret_guest, total_n, cancel_prob):
    # Returns up to 4 bullet points. Each is direction and text where direction is 'up', 'down', or 'neut'
    room_rate = avg_usd / EUR_TO_USD
    drivers = []

    if lead_t > 100:
        drivers.append(("up", "Long lead time — more time for plans to change"))
    elif lead_t < 14:
        drivers.append(("down", "Booked close to arrival — less window to cancel"))
    else:
        drivers.append(("neut", "Lead time is fairly normal, not a strong signal either way"))

    if special_req == 0:
        drivers.append(("up", "No special requests — guest hasn't invested much in the stay"))
    elif special_req >= 2:
        drivers.append(("down", f"{special_req} special requests — guest seems more committed"))
    else:
        drivers.append(("neut", "One special request — minor positive signal"))

    if prev_cancel >= 1:
        drivers.append(("up", f"{prev_cancel} prior cancellation on record — adds some risk"))
    elif ret_guest == "Yes":
        drivers.append(("down", "Returning guest, these guests tend to show up"))

    if bk_type == "Online":
        drivers.append(("up", "Online booking  have higher cancel rates overall"))
    elif bk_type in ["Corporate", "Offline"]:
        drivers.append(("down", f"{bk_type} booking tends to be more reliable"))

    if room_rate > 200:
        drivers.append(("up", "High room price causes bookings cancel more often"))
    elif room_rate < 70:
        drivers.append(("neut", "Low room price do flexible booking so not a strong risk signal"))

    return drivers[:4]


def get_impact_text(feature_label, raw_value):
    """Short label for the summary table"""
    try:
        if feature_label == "Days before arrival booking was made":
            v = int(float(raw_value))
            if v > 100:
                return "Increases the cancellation risk"
            elif v <= 14:
                return "Reduces the cancellation risk"
            else:
                return "Moderate effect"

        elif feature_label == "Special requests count":
            v = int(float(raw_value))
            if v == 0:
                return "Increases the cancellation risk"
            elif v >= 2:
                return "Reduces the cancellation risk"
            else:
                return "Slight positive effect"

        elif feature_label == "Previous cancellations":
            v = int(float(raw_value))
            if v >= 2:
                return "Significantly increases the risk"
            elif v == 1:
                return "Slightly increases the risk"
            else:
                return "No prior cancellation history"

        elif feature_label == "Returning guest":
            if raw_value == "Yes":
                return "Reduces cancellation risk"
            else:
                return "First time guest"

        elif feature_label == "Booking type":
            if raw_value == "Online":
                return "Slight increase in risk"
            elif raw_value in ["Corporate", "Offline"]:
                return "Slight reduction in risk"
            else:
                return "Neutral"

        elif feature_label == "Average room price per night (USD)":
            v = float(raw_value.replace("$", "").replace(",", ""))
            eur = v / EUR_TO_USD
            if eur > 200:
                return "High price may increases the risk"
            elif eur < 70:
                return "Low price can cause minor the effect"
            else:
                return "Moderate pricing almost neutral"

        elif feature_label == "Successful previous bookings":
            v = int(float(raw_value))
            if v >= 3:
                return "Strong history reduces the risk"
            elif v >= 1:
                return "Some positive history"
            else:
                return "No prior booking history"

        elif feature_label in ("Weekend nights booked", "Weekday nights booked"):
            v = int(float(raw_value))
            if v == 0:
                return "Zero nights, unusual booking"
            else:
                return "Normal stay duration"

        elif feature_label == "Adults staying":
            v = int(float(raw_value))
            if v == 0:
                return "No adults, unusual booking"
            else:
                return "Standard guest count"

        else:
            return "—"

    except Exception:
        return "—"


def render_feature_chart(top_feats):
    labels = top_feats.index.tolist()[::-1]
    values = top_feats.values.tolist()[::-1]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(7, max(2.5, n * 0.42)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # highlight the top bar
    bar_colors = ["#2563EB" if i == n - 1 else "#93c5fd" for i in range(n)]
    ax.barh(labels, values, color=bar_colors, height=0.58, edgecolor="none")

    ax.set_xlabel("Relative Importance", fontsize=8.5, color="#6b7280", labelpad=6)
    ax.tick_params(axis="y", labelsize=8.5, colors="#374151", pad=4)
    ax.tick_params(axis="x", labelsize=7.5, colors="#9ca3af")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#e5e7eb")
    ax.grid(axis="x", color="#f3f4f6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for i, (val, label) in enumerate(zip(values, labels)):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=7.5, color="#6b7280")

    plt.tight_layout(pad=0.6)
    return fig


# First load is slow i.e ~20s, because it trains on 36k rows 
with st.spinner("Training model,this takes about 20 seconds on first load..."):
    model = train_model()

top_importances = get_top_importances(model)


with st.sidebar:
    st.header("Input Details")
    st.write("Enter the booking details below to generate theprediction.")

    st.subheader("Guest Details")
    no_adults = st.number_input("Adults staying", 1, 10, 2,
                                help="Number of adults included in the booking")
    no_children = st.number_input("Children staying", 0, 10, 0,
                                  help="Number of children included in the booking")
    repeated_guest = st.selectbox("Returning guest", ["No", "Yes"],
                                  help="Whether the guest has stayed at this hotel before")
    prev_cancels = st.number_input("Previous cancellations", 0, 13, 0,
                                   help="Number of prior bookings this guest has cancelled")
    prev_ok = st.number_input("Successful previous bookings", 0, 58, 0,
                              help="Previous bookings completed without cancellation")
    special_req = st.slider("Special requests count", 0, 5, 0,
                            help="Number of special requests made alongside the booking")

    st.subheader("Stay Details")
    weekend_nights = st.number_input("Weekend nights booked", 0, 7, 1,
                                     help="Number of weekend nights (Saturday or Sunday)")
    week_nights = st.number_input("Weekday nights booked", 0, 17, 2,
                                  help="Number of weekday nights (Monday to Friday)")
    meal_display = st.selectbox("Meal plan selected", MEAL_DISPLAY_OPTIONS,
                                help="Meal arrangement included in the reservation")
    room_display = st.selectbox("Room type booked", ROOM_DISPLAY_OPTIONS,
                                help="Category of room reserved")
    car_parking = st.selectbox("Car parking required", ["No", "Yes"],
                               help="Whether the guest has requested a parking space")
    avg_price_usd = st.number_input("Average room price per night (USD)",
                                    0.0, 660.0, 110.0, step=5.0,
                                    help="Average nightly room rate in US dollars")

    st.subheader("Booking and Arrival")
    lead_time = st.number_input("Days before arrival booking was made", 0, 500, 30,
                                help="Days between booking date and arrival date")
    booking_type = st.selectbox("Booking type", SEGMENT_OPTIONS,
                                help="Channel through which the reservation was made")
    arrival_month = st.selectbox("Arrival month", list(MONTH_NAMES.keys()),
                                 format_func=lambda x: MONTH_NAMES[x], index=5,
                                 help="Month the guest is expected to arrive")
    arrival_date = st.number_input("Arrival day of month", 1, 31, 15,
                                   help="Day of the month the guest is expected to arrive")

    st.divider()
    predict_button = st.button("Generate Prediction")


st.title("Hotel Cancellation Predictor")
st.write("Predicts whether a hotel reservation is likely to be cancelled and is based on booking details and the behaviour of the guest.")

st.divider()


if predict_button:
    total_nights_val = int(weekend_nights) + int(week_nights)
    total_guests_val = int(no_adults) + int(no_children)

    if total_guests_val < 1:
        st.warning("Total guests is zero and valid booking requires at least one adult.")
    if total_nights_val < 1:
        st.warning("Total nights is zero so, please enter at least one night for a meaningful prediction.")

    meal_internal = meal_map[meal_display]
    room_internal = room_map[room_display]
    rg_val = 1 if repeated_guest == "Yes" else 0
    cp_val = 1 if car_parking == "Yes" else 0
    # Doing the Sanity check to avoid div by zero even though it shouldn't happen with a fixed constant
    avg_price_eur = avg_price_usd / EUR_TO_USD if EUR_TO_USD else avg_price_usd

    input_df = pd.DataFrame([{
        "type_of_meal_plan": meal_internal,
        "room_type_reserved": room_internal,
        "market_segment_type": booking_type,
        "no_of_adults": float(no_adults),
        "no_of_children": float(no_children),
        "no_of_weekend_nights": float(weekend_nights),
        "no_of_week_nights": float(week_nights),
        "required_car_parking_space": float(cp_val),
        "lead_time": float(lead_time),
        "arrival_month": float(arrival_month),
        "arrival_date": float(arrival_date),
        "repeated_guest": float(rg_val),
        "no_of_previous_cancellations": float(prev_cancels),
        "no_of_previous_bookings_not_canceled": float(prev_ok),
        "avg_price_per_room": float(avg_price_eur),
        "no_of_special_requests": float(special_req),
        "total_nights": float(total_nights_val),
        "total_guests": float(total_guests_val),
    }])

    proba = model.predict_proba(input_df)[0]
    cancel_p = round(proba[1] * 100, 1)
    proceed_p = round(proba[0] * 100, 1)
    predicted_cancel = proba[1] >= CANCEL_THRESHOLD

    if predicted_cancel:
        st.error(f"**Reservation Shows Elevated Cancellation Risk**  \n"
                 f"The model estimates a **{cancel_p}%** probability of cancellation. "
                 f"This booking profile is consistent with reservations that have been cancelled.")
    else:
        st.success(f"**Booking Shows Low Risk of Cancellation**  \n"
                   f"The model estimates a **{proceed_p}%** probability that the booking will be honoured. "
                   f"This profile is consistent with the reservations that proceed to check-in.")

    st.subheader("Confidence Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Booking Proceeds", f"{proceed_p}%")
        st.progress(int(proceed_p))
    with col2:
        st.metric("Cancelled", f"{cancel_p}%")
        st.progress(int(cancel_p))

    st.subheader("Key Drivers of This Prediction")

    key_drivers = get_key_drivers(
        int(lead_time), int(special_req), avg_price_usd,
        int(prev_cancels), booking_type, repeated_guest,
        total_nights_val, proba[1],
    )

    for direction, text in key_drivers:
        if direction == "up":
            st.write(f"- Increased risk: {text}")
        elif direction == "down":
            st.write(f"- Reduced risk: {text}")
        else:
            st.write(f"- Neutral: {text}")

    st.subheader("Key Factors Influencing This Prediction")
    st.caption("Features ranked by relative importance — shows what the model relies on most when assessing cancellation risk.")

    fig = render_feature_chart(top_importances)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.subheader("Explanation")
    explanation = build_explanation(
        int(lead_time), int(special_req), avg_price_usd,
        int(prev_cancels), booking_type, repeated_guest,
        total_nights_val, proba[1],
    )
    st.info(explanation)

    st.subheader("Summary of Inputs Used")

    summary_rows = [
        ("Adults staying",                       str(int(no_adults)),        get_impact_text("Adults staying", str(no_adults))),
        ("Children staying",                     str(int(no_children)),      get_impact_text("Children staying", str(no_children))),
        ("Weekend nights booked",                str(int(weekend_nights)),   get_impact_text("Weekend nights booked", str(weekend_nights))),
        ("Weekday nights booked",                str(int(week_nights)),      get_impact_text("Weekday nights booked", str(week_nights))),
        ("Meal plan selected",                   meal_display,               "—"),
        ("Car parking required",                 car_parking,                "—"),
        ("Room type booked",                     room_display,               "—"),
        ("Days before arrival booking was made", str(int(lead_time)),        get_impact_text("Days before arrival booking was made", str(lead_time))),
        ("Arrival month",                        MONTH_NAMES[arrival_month], "—"),
        ("Arrival day of month",                 str(int(arrival_date)),     "—"),
        ("Booking type",                         booking_type,               get_impact_text("Booking type", booking_type)),
        ("Returning guest",                      repeated_guest,             get_impact_text("Returning guest", repeated_guest)),
        ("Previous cancellations",               str(int(prev_cancels)),     get_impact_text("Previous cancellations", str(prev_cancels))),
        ("Successful previous bookings",         str(int(prev_ok)),          get_impact_text("Successful previous bookings", str(prev_ok))),
        ("Average room price per night (USD)",   f"${avg_price_usd:,.2f}",   get_impact_text("Average room price per night (USD)", f"${avg_price_usd:,.2f}")),
        ("Special requests count",               str(int(special_req)),      get_impact_text("Special requests count", str(special_req))),
    ]

    summary_df = pd.DataFrame(summary_rows, columns=["Feature", "Value Entered", "Impact on Prediction"])
    st.table(summary_df)

else:
    st.info("Complete  the necessary fields which apperas in the sidebar and click Generate Prediction.")


st.divider()
with st.expander("How this works"):
    st.markdown("""
**Model**

This tool the tuned Random Forest classifier which is trained on the historical hotel
reservation data. It was chosen after the testing it against the Logistic Regression
and the Decision Tree baselines. The training set includes nealrly 36,000 bookings
with known outcomes.

**Class imbalance and threshold**

Almost the third of the bookings in the training data were cancelled roughly 33%.
To stop the model from leaning too heavily toward the majority class, class
weights are applied during the training. The prediction threshold is also set to
0.40 instead of the default 0.50.

That means if a booking is flagged as at risk when the model estimates a
cancellation probability of 40% or bit or much higher. This makes the tool more
crucial to likely cancellations, which are usually more costly to miss.

**Training data**

Only bookings with a confirmed outcome cancelled or completed were used
for training. The arrival year field was removed because the dataset only
covers the two years which are 2017 and 2018, so it would not be useful outside that period.

**Features used**

The model uses important information such as guest composition, stay length, room type,
meal plan, booking channel, price, lead time, and previous cancellation
history. Lead time and past cancellation behaviour considers to be the strongest signals.

**Limitations**

This model cannot predict cancellations with complete certainty. Its probability
estimates are based on historical patterns and it may not fully reflect the current
booking conditions which are changing time to time . It should be used as a decision support tool for revenue
management teams and not as a replacement for operational judgement.
""")


