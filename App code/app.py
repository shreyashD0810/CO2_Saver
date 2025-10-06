import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from keras.models import load_model

# Page config
st.set_page_config(page_title="CO₂ Emissions Dashboard", layout="wide")
st.title("🌍 Global CO₂ Emissions Dashboard")

# === Load all data ===
@st.cache_data
def load_data():
    data = {
        "co2": pd.read_csv("cleaned_co2_data.csv"),
        "gdp": pd.read_csv("co2_per_gdp_latest.csv"),
        "country_total": pd.read_csv("country_wise_total_emissions.csv"),
        "top_emitters": pd.read_csv("top_10_emitters_latest_year.csv"),
        "sector_2023": pd.read_csv("sector_wise_contribution_latest_year.csv"),
        "sector_all": pd.read_csv("sector_wise_all_time.csv")
    }
    return data

data = load_data()

# Define pages
TABS = [
    "Choropleth Map",
    "Sector-wise Analysis",
    "Country-wise Emissions",
    "Top Emitters",
    "CO₂ vs GDP",
    "LSTM Forecast"
]

# Initialize session state
if "active_tab" not in st.session_state:
    st.session_state.active_tab = TABS[0]

# Sidebar UI
st.sidebar.markdown("## Navigation")
for tab in TABS:
    if st.sidebar.button(tab, use_container_width=True, key=tab):
        st.session_state.active_tab = tab

# Styling (optional)
st.markdown(
    """
    <style>
    button[kind="secondary"] {
        background-color: #1e293b !important;
        color: white !important;
        border-radius: 8px;
        margin-bottom: 6px;
    }
    button[kind="secondary"]:hover {
        background-color: #2563eb !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load selected page
page = st.session_state.active_tab
st.title(f"📊 {page}")



# === Page: Choropleth Map ===
if page == "Choropleth Map":
    st.header("🌐 Global Emissions by Year")

    selected_year = st.slider(
        "Select Year",
        int(data["co2"].year.min()),
        int(data["co2"].year.max()),
        2022
    )

    df_year = data["co2"][data["co2"].year == selected_year]

    # Define custom harsh color scale (low = light blue, high = deep navy)
    harsh_blue_scale = [
        [0.0, "#d4f0ff"],   # very light blue
        [0.2, "#86c5e5"],
        [0.4, "#4f9edc"],
        [0.6, "#1f78b4"],
        [0.8, "#08306b"],
        [1.0, "#041c40"]    # very dark blue / navy
    ]

    fig = px.choropleth(
        df_year,
        locations="country",
        locationmode="country names",
        color="co2",
        hover_name="country",
        color_continuous_scale=harsh_blue_scale,
        title=f"CO₂ Emissions in {selected_year}"
    )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title="CO₂ Emissions (Mt)",
            ticksuffix=" Mt",
            lenmode="fraction",
            len=0.75
        )
    )

    st.plotly_chart(fig, use_container_width=True)


# === Page: Sector-wise Analysis ===
elif page == "Sector-wise Analysis":
    st.header("🏭 Sector-wise Emissions")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Latest Year Contribution")
        fig1 = px.pie(
            data["sector_2023"],
            names="sector",
            values="co2_emissions",
            title="CO₂ by Sector (Latest Year)"
        )
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.subheader("All-time Sector Totals")
        fig2 = px.bar(
            data["sector_all"],
            x="sector",
            y="total_emissions",
            title="All-time CO₂ by Sector"
        )
        st.plotly_chart(fig2, use_container_width=True)

# === Page: Country-wise Emissions ===
elif page == "Country-wise Emissions":
    st.header("📈 Country Total Emissions")
    top_n = st.slider("Top N Countries", 5, 50, 15)
    df_ct = data["country_total"].sort_values("co2", ascending=False).head(top_n)
    fig3 = px.bar(
        df_ct,
        x="country",
        y="co2",
        title="Top Countries by Total CO₂ Emissions"
    )
    st.plotly_chart(fig3, use_container_width=True)

# === Page: Top Emitters ===
elif page == "Top Emitters":
    st.header("🏆 Top Emitters (Latest Year)")
    df_te = data["top_emitters"].rename(columns={"co2": "co2_emissions"})
    fig4 = px.bar(
        df_te,
        x="country",
        y="co2_emissions",
        title="Top 10 CO₂ Emitters"
    )
    st.plotly_chart(fig4, use_container_width=True)

# === Page: CO2 vs GDP ===
elif page == "CO₂ vs GDP":
    st.header("💸 CO₂ Efficiency vs GDP (Latest Year)")
    # Merge CO2 per GDP with total emissions to compute GDP
    df_gdp = data["gdp"].merge(
        data["country_total"].loc[:, ["country", "co2"]].rename(columns={"co2": "co2_emissions"}),
        on="country"
    )
    df_gdp["year"] = df_gdp["year_x"] if "year_x" in df_gdp else df_gdp["year"]
    df_gdp["gdp"] = df_gdp["co2_emissions"] / df_gdp["co2_per_gdp"]
    df_gdp.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_gdp.dropna(subset=["gdp"], inplace=True)
    fig5 = px.scatter(
        df_gdp,
        x="gdp",
        y="co2_per_gdp",
        size="co2_per_gdp",
        hover_name="country",
        log_x=True,
        title="CO₂ per GDP vs Total GDP"
    )
    st.plotly_chart(fig5, use_container_width=True)

# === Page: LSTM Forecast ===
elif page == "LSTM Forecast":
    st.header("📉 CO₂ Forecast with LSTM")
    countries = sorted(data["co2"].country.unique())
    country = st.selectbox("Select Country", countries)
    df_c = data["co2"][data["co2"].country == country].sort_values("year")
    series = df_c.co2.values.reshape(-1, 1)
    try:
        scaler = joblib.load("scaler.pkl")
        model = load_model("lstm_model.keras")
    except Exception:
        st.error("Model or scaler not found. Please train and save them first.")
        st.stop()
    scaled = scaler.transform(series)
    n_steps = 5
    # Forecast last sequence
    seq = scaled[-n_steps:]
    preds = []
    for _ in range(10):
        pred = model.predict(seq.reshape(1, n_steps, 1), verbose=0)
        preds.append(pred[0][0])
        seq = np.vstack((seq[1:], pred))
    years = list(range(int(df_c.year.max())+1, int(df_c.year.max())+11))
    forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    df_hist = df_c[["year", "co2"]]
    df_pred = pd.DataFrame({"year": years, "co2": forecast})
    df_plot = pd.concat([df_hist, df_pred])
    fig6 = px.line(
        df_plot,
        x="year",
        y="co2",
        title=f"CO₂ Emissions Forecast for {country}",
        markers=True
    )
    st.plotly_chart(fig6, use_container_width=True)

