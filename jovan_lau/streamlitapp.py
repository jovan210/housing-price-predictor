import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).parent


st.set_page_config(
    page_title="Kingsmen | King County Valuations",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@300;400;500;600&display=swap');

/* hide Streamlit chrome */
#MainMenu, header, footer {visibility: hidden;}
div[data-testid="stToolbar"] {visibility: hidden; height: 0;}
div[data-testid="stDecoration"] {display: none;}

/* page background */
.stApp {
    background: linear-gradient(160deg, #071324 0%, #0C2038 55%, #102A47 100%);
    color: #E8EEF6;
    font-family: 'Inter', sans-serif;
}

/* sidebar */
section[data-testid="stSidebar"] > div {
    background: #050E1B;
    border-right: 1px solid rgba(198,161,91,0.25);
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem !important;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #C6A15B !important;
    border-bottom: 1px solid rgba(198,161,91,0.2);
    padding-bottom: 0.4rem;
    margin-top: 1.4rem;
}
section[data-testid="stSidebar"] label {
    color: #A9BBD0 !important;
    font-size: 0.84rem !important;
}

/* headings */
h1 {
    font-family: 'Fraunces', serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    color: #F4F7FB !important;
}
h2, h3 { font-family: 'Fraunces', serif !important; color: #F4F7FB !important; }

/* buttons */
.stButton > button {
    background: #C6A15B;
    color: #071324;
    border: none;
    border-radius: 2px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 0.78rem;
    padding: 0.6rem 1rem;
}
.stButton > button:hover { background: #D9B978; color: #071324; }

/* download button */
.stDownloadButton > button {
    background: transparent;
    color: #C6A15B;
    border: 1px solid #C6A15B;
    border-radius: 2px;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* the headline valuation panel */
.valuation {
    background: rgba(255,255,255,0.035);
    border-left: 3px solid #C6A15B;
    padding: 1.6rem 2rem;
    margin: 0.5rem 0 1.6rem 0;
}
.valuation .label {
    font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: #8FA4BD; margin-bottom: 0.4rem;
}
.valuation .figure {
    font-family: 'Fraunces', serif; font-size: 3.2rem; font-weight: 700;
    color: #F4F7FB; line-height: 1;
}
.valuation .range { font-size: 0.85rem; color: #8FA4BD; margin-top: 0.5rem; }

/* small stat blocks */
.stat { border-top: 1px solid rgba(198,161,91,0.25); padding-top: 0.6rem; }
.stat .k {
    font-size: 0.65rem; letter-spacing: 0.14em; text-transform: uppercase; color: #8FA4BD;
}
.stat .v { font-size: 1.05rem; color: #E8EEF6; font-weight: 500; }

/* dataframe */
div[data-testid="stDataFrame"] { border: 1px solid rgba(198,161,91,0.2); }
</style>
""", unsafe_allow_html=True)


MODEL_MAE = 68933      
MODEL_R2 = 0.886       

model = joblib.load(BASE_DIR / "model.pkl")
model_columns = joblib.load(BASE_DIR / "model_columns.pkl")
zip_lookup = pd.read_csv(BASE_DIR / "zipcode_lookup.csv")
zipcodes = sorted(zip_lookup['zipcode'].tolist())

@st.cache_data
def load_market_data():
    """Load the raw sales data used for the market comparison charts."""
    try:
        return pd.read_csv(BASE_DIR / "kc_house_data.csv")
    except FileNotFoundError:
        return None

market = load_market_data()

if "history" not in st.session_state:
    st.session_state.history = []


NAVY, BRASS, MIST = "#0C2038", "#C6A15B", "#8FA4BD"
plt.rcParams.update({
    "text.color":        "#E8EEF6",
    "axes.titlecolor":   "#E8EEF6",
    "axes.labelcolor":   MIST,
    "xtick.color":       MIST,
    "ytick.color":       MIST,
    "axes.edgecolor":    "#2C4463",
    "axes.facecolor":    "none",
    "figure.facecolor":  "none",
    "savefig.facecolor": "none",
    "font.size":         9,
})

def style_axes(ax):
    """Remove chart junk and keep the panel transparent."""
    ax.set_facecolor("none")
    ax.figure.patch.set_alpha(0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["bottom", "left"]:
        ax.spines[side].set_color("#2C4463")
    ax.tick_params(colors=MIST, labelsize=8)
    ax.xaxis.label.set_color(MIST)
    ax.yaxis.label.set_color(MIST)
    ax.title.set_color("#E8EEF6")


st.sidebar.markdown("### Location")
default_zip_index = zipcodes.index(98103) if 98103 in zipcodes else 0
zipcode_selected = st.sidebar.selectbox(
    "Zipcode", zipcodes, index=default_zip_index,
    help="Neighbourhood is one of the strongest drivers of price in King County."
)
waterfront_selected = st.sidebar.checkbox("Waterfront property")
view_selected = st.sidebar.slider("View rating", 0, 4, 0,
                                  help="0 = no notable view, 4 = excellent.")

st.sidebar.markdown("### Dimensions")
c1, c2 = st.sidebar.columns(2)
bedrooms_selected = c1.number_input("Bedrooms", 1, 10, 3)
bathrooms_selected = c2.number_input("Bathrooms", 0.5, 8.0, 2.0, step=0.25,
    help="US listings count part-baths: 0.5 = toilet and sink, 0.75 adds a shower.")
sqft_living_selected = st.sidebar.slider("Living area (sqft)", 300, 10000, 2000, step=50)
sqft_lot_selected = st.sidebar.slider("Lot size (sqft)", 500, 100000, 7500, step=500)
sqft_basement_selected = st.sidebar.slider("Basement (sqft)", 0, 3000, 0, step=50,
    help="Leave at 0 if there is no basement.")
floors_selected = st.sidebar.slider("Floors", 1.0, 3.5, 1.0, step=0.5)

st.sidebar.markdown("### Condition")
grade_selected = st.sidebar.select_slider("Construction grade", options=list(range(3, 14)), value=7,
    help="King County build-quality scale: 3-6 below average, 7 average, 11-13 luxury.")
condition_selected = st.sidebar.select_slider("Condition", options=[1, 2, 3, 4, 5], value=3,
    help="1 = poor, 3 = average, 5 = excellent.")

st.sidebar.markdown("### History")
yr_built_selected = st.sidebar.slider("Year built", 1900, 2015, 1975)
yr_renovated_selected = st.sidebar.number_input("Year renovated", 0, 2015, 0,
    help="Leave at 0 if never renovated.")

predict_clicked = st.sidebar.button("Value this property", use_container_width=True)


st.markdown(
    "<div style='font-size:0.7rem;letter-spacing:0.28em;text-transform:uppercase;"
    "color:#C6A15B;margin-bottom:0.2rem;'>Kingsmen Property Analytics</div>",
    unsafe_allow_html=True
)
st.title("King County Valuations")
st.markdown(
    "<p style='color:#8FA4BD;max-width:60ch;'>A random forest model trained on "
    "21,000 recorded sales across King County, Washington. Enter the property details "
    "on the left to generate an estimated market value.</p>",
    unsafe_allow_html=True
)


if predict_clicked:

    if sqft_basement_selected >= sqft_living_selected:
        st.error("Basement area must be smaller than the total living area. Please adjust your inputs.")

    elif yr_renovated_selected != 0 and yr_renovated_selected < yr_built_selected:
        st.error(f"Renovation year cannot be earlier than the year built ({yr_built_selected}).")

    else:

        zip_row = zip_lookup[zip_lookup['zipcode'] == zipcode_selected].iloc[0]

        sqft_above = sqft_living_selected - sqft_basement_selected


        house_age = 2015 - yr_built_selected
        is_renovated = 1 if yr_renovated_selected > 0 else 0

        df_input = pd.DataFrame({
            'bedrooms': [bedrooms_selected], 'bathrooms': [bathrooms_selected],
            'sqft_living': [sqft_living_selected], 'sqft_lot': [sqft_lot_selected],
            'floors': [floors_selected], 'waterfront': [int(waterfront_selected)],
            'view': [view_selected], 'condition': [condition_selected],
            'grade': [grade_selected], 'sqft_above': [sqft_above],
            'sqft_basement': [sqft_basement_selected], 'yr_built': [yr_built_selected],
            'yr_renovated': [yr_renovated_selected], 'zipcode': [str(zipcode_selected)],
            'lat': [zip_row['lat']], 'long': [zip_row['long']],
            'sqft_living15': [zip_row['sqft_living15']], 'sqft_lot15': [zip_row['sqft_lot15']],
            'house_age': [house_age], 'is_renovated': [is_renovated]
        })


        df_input = pd.get_dummies(df_input, columns=['zipcode'])
        df_input = df_input.reindex(columns=model_columns, fill_value=0)

        price = model.predict(df_input)[0]
        margin = MODEL_MAE

        st.markdown(f"""
        <div class="valuation">
            <div class="label">Estimated market value</div>
            <div class="figure">${price:,.0f}</div>
            <div class="range">Typical range ${price-margin:,.0f} &nbsp;—&nbsp; ${price+margin:,.0f}
            &nbsp;·&nbsp; based on a mean absolute error of ${margin:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        for col, k, v in [
            (s1, "Price per sqft", f"${price/sqft_living_selected:,.0f}"),
            (s2, "Zipcode", f"{zipcode_selected}"),
            (s3, "Configuration", f"{bedrooms_selected} bd · {bathrooms_selected} ba"),
            (s4, "Built", f"{yr_built_selected} · grade {grade_selected}")
        ]:
            col.markdown(f"<div class='stat'><div class='k'>{k}</div><div class='v'>{v}</div></div>",
                         unsafe_allow_html=True)

        st.session_state.history.append({
            "zipcode": zipcode_selected, "bedrooms": bedrooms_selected,
            "bathrooms": bathrooms_selected, "sqft_living": sqft_living_selected,
            "sqft_lot": sqft_lot_selected, "floors": floors_selected,
            "basement": sqft_basement_selected, "grade": grade_selected,
            "condition": condition_selected, "yr_built": yr_built_selected,
            "yr_renovated": yr_renovated_selected, "waterfront": int(waterfront_selected),
            "view": view_selected, "predicted_price": round(price, 2)
        })

        local = market[market['zipcode'] == zipcode_selected] if market is not None else None

        if local is not None and len(local) > 0:
            st.markdown("### Market context")
            g1, g2 = st.columns(2)

            with g1:
                fig, ax = plt.subplots(figsize=(5, 3.1))
                ax.hist(local['price'], bins=35, color="#274866", edgecolor="none")
                ax.axvline(price, color=BRASS, linewidth=2)
                ax.set_title(f"Price distribution · {zipcode_selected}", fontsize=10, loc="left")
                ax.set_xlabel("Sale price ($)")
                ax.set_ylabel("Properties sold")
                ax.xaxis.set_major_formatter(
                    plt.FuncFormatter(lambda v, p: f"{int(v/1000)}k"))
                style_axes(ax)
                plt.tight_layout()
                st.pyplot(fig, transparent=True)
                plt.close(fig)
                pct = (local['price'] < price).mean() * 100
                st.caption(f"This valuation sits above {pct:.0f}% of recorded sales in {zipcode_selected}.")

            with g2:
                fig, ax = plt.subplots(figsize=(5, 3.1))
                ax.scatter(local['sqft_living'], local['price'], s=8,
                           color="#274866", alpha=0.7, edgecolors="none")
                ax.scatter([sqft_living_selected], [price], s=90, color=BRASS,
                           marker="D", zorder=5, label="This property")
                ax.set_title(f"Living area against price · {zipcode_selected}", fontsize=10, loc="left")
                ax.set_xlabel("Living area (sqft)")
                ax.set_ylabel("Sale price ($)")
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda v, p: f"{int(v/1000)}k"))
                leg = ax.legend(frameon=False, fontsize=8)
                for t in leg.get_texts():
                    t.set_color(MIST)
                style_axes(ax)
                plt.tight_layout()
                st.pyplot(fig, transparent=True)
                plt.close(fig)

                comps = local[
                    local['sqft_living'].between(sqft_living_selected*0.9, sqft_living_selected*1.1)
                ]
                if len(comps) >= 3:
                    st.caption(
                        f"{len(comps)} comparable sales of similar size in this zipcode, "
                        f"median ${comps['price'].median():,.0f}."
                    )


if st.session_state.history:
    st.markdown("### Valuation log")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

    e1, e2 = st.columns([1, 4])
    with e1:
        st.download_button(
            "Download CSV",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name="kingsmen_valuations.csv",
            mime="text/csv",
            use_container_width=True
        )
    with e2:
        if st.button("Clear log"):
            st.session_state.history = []
            st.rerun()

elif not predict_clicked:
    st.markdown(
        "<div style='border:1px dashed rgba(198,161,91,0.35);padding:2rem;"
        "text-align:center;color:#8FA4BD;margin-top:1rem;'>"
        "Enter property details in the sidebar and select "
        "<span style='color:#C6A15B;'>Value this property</span> to begin."
        "</div>", unsafe_allow_html=True
    )


st.markdown(
    "<hr style='border-color:rgba(198,161,91,0.2);margin-top:2.5rem;'>"
    "<p style='color:#6C82A0;font-size:0.78rem;max-width:75ch;'>"
    f"Model: random forest regressor, R² {MODEL_R2:.3f} on held-out data, mean absolute "
    f"error ${MODEL_MAE:,}. Estimates are most reliable for realistic input combinations "
    "— bedroom count and living area normally increase together. This is a guide price, "
    "not a formal valuation.</p>",
    unsafe_allow_html=True
)