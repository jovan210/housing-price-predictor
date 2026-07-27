import joblib
import streamlit as st
import pandas as pd

## Page configuration - sets browser tab title and uses the full window width
st.set_page_config(page_title="KC House Price Predictor", page_icon="🏠", layout="wide")

## Load trained model and the zipcode lookup table
model = joblib.load("kc_best_gbt_model.pkl")
zip_lookup = pd.read_csv("zipcode_lookup.csv")

zipcodes = sorted(zip_lookup['zipcode'].tolist())

## ---------------- SIDEBAR: all user inputs ----------------
## Putting inputs in the sidebar keeps the result visible without scrolling
st.sidebar.header("Property details")

st.sidebar.subheader("Location")
zipcode_selected = st.sidebar.selectbox(
    "Zipcode", zipcodes, index=zipcodes.index(98103),
    help="The neighbourhood is one of the strongest drivers of price in King County."
)
waterfront_selected = st.sidebar.checkbox("Waterfront property")
view_selected = st.sidebar.slider(
    "View rating", 0, 4, 0,
    help="0 = no notable view, 4 = excellent view."
)

st.sidebar.subheader("Size")
col_a, col_b = st.sidebar.columns(2)
bedrooms_selected = col_a.number_input("Bedrooms", 1, 10, 3)
bathrooms_selected = col_b.number_input(
    "Bathrooms", 0.5, 8.0, 2.0, step=0.25,
    help="US listings count part-baths: 0.5 = toilet and sink, 0.75 = adds a shower."
)
sqft_living_selected = st.sidebar.slider("Living area (sqft)", 300, 10000, 2000, step=50)
sqft_lot_selected = st.sidebar.slider("Lot size (sqft)", 500, 100000, 7500, step=500)
sqft_basement_selected = st.sidebar.slider("Basement area (sqft)", 0, 3000, 0, step=50,
                                           help="Set to 0 if the house has no basement.")
floors_selected = st.sidebar.slider("Floors", 1.0, 3.5, 1.0, step=0.5)

st.sidebar.subheader("Quality")
grade_selected = st.sidebar.select_slider(
    "Construction grade", options=list(range(3, 14)), value=7,
    help="King County's build-quality scale: 3-6 below average, 7 average, 11-13 luxury."
)
condition_selected = st.sidebar.select_slider(
    "Condition", options=[1, 2, 3, 4, 5], value=3,
    help="1 = poor and needs work, 3 = average, 5 = excellent."
)

st.sidebar.subheader("History")
yr_built_selected = st.sidebar.slider("Year built", 1900, 2015, 1975)
yr_renovated_selected = st.sidebar.number_input(
    "Year renovated", min_value=0, max_value=2015, value=0,
    help="Leave as 0 if the property has never been renovated."
)

predict_clicked = st.sidebar.button("Predict house price", type="primary", use_container_width=True)

## ---------------- MAIN AREA: title and result ----------------
st.title("🏠 King County House Price Predictor")
st.write(
    "Estimate the market value of a house in King County, Washington. "
    "Set the property details in the sidebar, then click **Predict house price**."
)

if predict_clicked:

    ## Input validation - basement cannot exceed total living area
    if sqft_basement_selected >= sqft_living_selected:
        st.error("Basement area must be smaller than total living area. Please adjust your inputs.")

    ## Input validation - renovation cannot happen before the house was built
    elif yr_renovated_selected != 0 and yr_renovated_selected < yr_built_selected:
        st.error(f"Renovation year cannot be earlier than the year built ({yr_built_selected}).")

    else:
        ## Look up neighbourhood features for the selected zipcode.
        ## A user knows their zipcode but not their latitude or their neighbours'
        ## average house size, so these are filled with the zipcode median.
        zip_row = zip_lookup[zip_lookup['zipcode'] == zipcode_selected].iloc[0]

        ## Derive the remaining features the same way they were derived in training
        sqft_above = sqft_living_selected - sqft_basement_selected
        house_age = 2015 - yr_built_selected
        is_renovated = 1 if yr_renovated_selected > 0 else 0

        ## Convert input data to a DataFrame
        df_input = pd.DataFrame({
            'bedrooms': [bedrooms_selected],
            'bathrooms': [bathrooms_selected],
            'sqft_living': [sqft_living_selected],
            'sqft_lot': [sqft_lot_selected],
            'floors': [floors_selected],
            'waterfront': [int(waterfront_selected)],
            'view': [view_selected],
            'condition': [condition_selected],
            'grade': [grade_selected],
            'sqft_above': [sqft_above],
            'sqft_basement': [sqft_basement_selected],
            'yr_built': [yr_built_selected],
            'yr_renovated': [yr_renovated_selected],
            'zipcode': [str(zipcode_selected)],
            'lat': [zip_row['lat']],
            'long': [zip_row['long']],
            'sqft_living15': [zip_row['sqft_living15']],
            'sqft_lot15': [zip_row['sqft_lot15']],
            'house_age': [house_age],
            'is_renovated': [is_renovated]
        })

        ## One-hot encoding - must mirror the training pipeline exactly
        df_input = pd.get_dummies(df_input, columns=['zipcode'])

        ## Add every column the model expects (missing zipcode dummies become 0)
        ## and put them in the exact order used during training
        df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

        ## Predict
        predicted_price = model.predict(df_input)[0]

        ## Show the result prominently, with the typical error as context
        st.metric(
            label="Estimated market value",
            value=f"${predicted_price:,.0f}",
            delta=f"± $79,000 typical error",
            delta_color="off"
        )

        ## Summarise the inputs so the user can see what the estimate is based on
        st.subheader("Based on")
        c1, c2, c3, c4 = st.columns(4)
        c1.write(f"**Zipcode**  \n{zipcode_selected}")
        c2.write(f"**Size**  \n{sqft_living_selected:,} sqft")
        c3.write(f"**Layout**  \n{bedrooms_selected} bed, {bathrooms_selected} bath")
        c4.write(f"**Built**  \n{yr_built_selected} (grade {grade_selected})")

        st.caption(
            "This model explains about 87% of price variation in King County, with a "
            "typical error of around $79,000. Estimates are most reliable for realistic "
            "combinations of inputs - for example, bedroom count and living area normally "
            "increase together. Treat this as a guide price, not a formal valuation."
        )

else:
    st.info("Set the property details in the sidebar, then click **Predict house price**.")