# ==========================
# IMPORTS
# ==========================
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================
# CONFIG PAGE
# ==========================
st.set_page_config(
    page_title="Prédiction Prix Voiture",
    page_icon="🚗",
    layout="centered"
)

# ==========================
# CHARGEMENT DES DONNÉES
# ==========================
df = pd.read_csv("Voitures.csv")  # ton dataset

X = df.drop("sale_price", axis=1)
y = df["sale_price"]

# ==========================
# SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# MODÈLE
# ==========================
model = LinearRegression()
model.fit(X_train, y_train)

# ==========================
# PERFORMANCE
# ==========================
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = np.sqrt(mean_squared_error(y_test, y_pred))

# ==========================
# INTERFACE
# ==========================
st.title("🚗 Prédiction du Prix d'une Voiture")
st.write("Veuillez renseigner les caractéristiques du véhicule :")

col1, col2 = st.columns(2)

with col1:
    age_voiture = st.number_input(
        "age_years",
        min_value=0,
        max_value=50,
        value=5
    )

    kilometrage = st.number_input(
        "mileage_km",
        min_value=0,
        max_value=500000,
        value=80000
    )


with col2:
    consommation = st.number_input(
        "fuel_type_Encoded",
        min_value=1.0,
        max_value=10.0,
        step=1.0
    )

    puissance = st.number_input(
        "engine_power_hp",
        min_value=50,
        max_value=500,
        value=120
    )

  
# ==========================
# PRÉDICTION
# ==========================
if st.button("🔮 Prédire le prix"):
    input_data = np.array([[age_voiture, kilometrage, puissance, consommation]])

    prediction = model.predict(input_data)
    prix = float(prediction[0])

    st.success(f"💰 Prix estimé de la voiture : **{prix:,.0f} €**")

# ==========================
# PERFORMANCE MODÈLE
# ==========================
st.subheader("📊 Performance du modèle")

col3, col4, col5 = st.columns(3)
col3.metric("R²", f"{r2:.3f}")
col4.metric("MAE (€)", f"{mae:,.0f}")
col5.metric("MSE (€)", f"{mse:,.0f}")

# ==========================
# INFO
# ==========================
st.info(
    "Ce modèle utilise une **Régression Linéaire** pour estimer le prix "
    "d’une voiture à partir de ses caractéristiques principales."
)
