import joblib
import streamlit as st
import pandas as pd

## Load trained model and the zipcode lookup table
model = joblib.load("kc_best_gbt_model.pkl")
zip_lookup = pd.read_csv("zipcode_lookup.csv")

## Streamlit app
st.title("King County House Price Predictor")
st.write("Estimate the market value of a house in King County, Washington.")

## Define the input options
zipcodes = sorted(zip_lookup['zipcode'].tolist())
conditions = [1, 2, 3, 4, 5]
grades = list(range(3, 14))

## User inputs - only features a homeowner or agent would actually know
zipcode_selected = st.selectbox("Zipcode", zipcodes, index=zipcodes.index(98103))
bedrooms_selected = st.slider("Bedrooms", 1, 10, 3)
bathrooms_selected = st.slider("Bathrooms", 0.5, 8.0, 2.0, step=0.25)
sqft_living_selected = st.slider("Living area (sqft)", 300, 10000, 2000, step=50)
sqft_lot_selected = st.slider("Lot size (sqft)", 500, 100000, 7500, step=500)
floors_selected = st.slider("Floors", 1.0, 3.5, 1.0, step=0.5)
sqft_basement_selected = st.slider("Basement area (sqft, 0 if none)", 0, 3000, 0, step=50)
grade_selected = st.selectbox("Construction grade (3 = poor, 13 = excellent)", grades, index=grades.index(7))
condition_selected = st.selectbox("Condition (1 = poor, 5 = excellent)", conditions, index=2)
yr_built_selected = st.slider("Year built", 1900, 2015, 1975)
yr_renovated_selected = st.number_input("Year renovated (0 if never)", min_value=0, max_value=2015, value=0)
waterfront_selected = st.checkbox("Waterfront property")
view_selected = st.slider("View rating (0 = none, 4 = excellent)", 0, 4, 0)

## Predict button
if st.button("Predict house price"):

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
        st.success(f"Predicted price: ${predicted_price:,.0f}")

        ## Give the user context for the number
        st.caption(
            "The model explains about 87% of price variation, with a typical error "
            "of around $79,000. Treat this as a guide price, not a valuation."
        )