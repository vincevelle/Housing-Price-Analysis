import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.express as px
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from folium.plugins import MarkerCluster

# Load dataset
california_housing = fetch_california_housing(as_frame=True)
df = california_housing.frame

# Feature Engineering
df['Rooms_per_Household'] = df['AveRooms'] / df['AveOccup']
df['Bedrooms_per_Room'] = df['AveBedrms'] / df['AveRooms']
df['Population_per_Household'] = df['Population'] / df['AveOccup']

# Streamlit UI Setup
st.title("California Housing Price Analysis")
st.sidebar.header("Filter Data")
price_range = st.sidebar.slider("Select Median House Value Range", float(df['MedHouseVal'].min()), float(df['MedHouseVal'].max()), (float(df['MedHouseVal'].min()), float(df['MedHouseVal'].max())))

# Cache the filtered dataset
@st.cache_data
def filter_data(df, price_range):
    return df[(df['MedHouseVal'] >= price_range[0]) & (df['MedHouseVal'] <= price_range[1])]

df_filtered = filter_data(df, price_range)

# Define features and target variable
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cache model training
@st.cache_resource
def train_models():
    with st.spinner("Training models... this may take a while ⏳"):
        # Train Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_test_pred = lr_model.predict(X_test_scaled)

        # Train Random Forest
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_test_pred = rf_model.predict(X_test_scaled)

        # Train Gradient Boosting
        gb_model = GradientBoostingRegressor(random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        gb_test_pred = gb_model.predict(X_test_scaled)

        st.success("Model training completed ✅")

    return lr_model, rf_model, gb_model, lr_test_pred, rf_test_pred, gb_test_pred

# Call the function once and reuse the results
lr_model, rf_model, gb_model, lr_test_pred, rf_test_pred, gb_test_pred = train_models()

# Cache the map generation
df_filtered = df_filtered.copy()  # Avoid modifying original dataframe
df_filtered["Predicted_Price"] = gb_model.predict(scaler.transform(df_filtered.drop(columns=["MedHouseVal"])))

@st.cache_resource
def create_map(df_filtered):
    map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=6)
    marker_cluster = MarkerCluster().add_to(map)

    for _, row in df_filtered.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color='blue',
            fill=True,
            fill_opacity=0.6,
            popup=f"Actual Price: ${row['MedHouseVal']*100000:.2f}\nPredicted Price: ${row['Predicted_Price']*100000:.2f}"
        ).add_to(marker_cluster)

    return map

# Generate the map
st.subheader("Comparison of Actual vs Predicted Home Prices in California")
map = create_map(df_filtered)
st_folium(map, width=700, height=500)
# Model Performance Comparison
st.subheader("Model Performance Comparison")
models = ["Linear Regression", "Random Forest", "Gradient Boosting"]
r2_scores = [r2_score(y_test, lr_test_pred), r2_score(y_test, rf_test_pred), r2_score(y_test, gb_test_pred)]
rmse_values = [np.sqrt(mean_squared_error(y_test, lr_test_pred)), np.sqrt(mean_squared_error(y_test, rf_test_pred)), np.sqrt(mean_squared_error(y_test, gb_test_pred))]

fig = px.bar(x=models, y=r2_scores, title="Model R² Score Comparison", labels={'x': "Model", 'y': "R² Score"})
st.plotly_chart(fig)

fig = px.bar(x=models, y=rmse_values, title="Model RMSE Comparison", labels={'x': "Model", 'y': "RMSE ($100,000)"})
st.plotly_chart(fig)

# Scatter Plot for Actual vs Predicted Prices
df_results = pd.DataFrame({'Actual': y_test, 'Linear Regression': lr_test_pred, 'Random Forest': rf_test_pred, 'Gradient Boosting': gb_test_pred})
st.subheader("Actual vs Predicted Prices")
fig = px.scatter(df_results, x='Actual', y='Gradient Boosting', title="Gradient Boosting Predictions vs Actual")
fig.add_shape(type="line", x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max(), line=dict(color='black', dash='dash'))
st.plotly_chart(fig)

st.write("This interactive interface allows users to explore actual and predicted home prices in California, compare different ML models, and visualize price predictions on a map.")
