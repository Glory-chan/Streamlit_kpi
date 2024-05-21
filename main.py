import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Titre de l'application
st.title("Kondo Godglory ewoènam")
st.title("TP KPI")

# Chargement des données
df = pd.read_csv('atomic_data.csv')

st.write("L'ndicateur utilisé pour connaître la santé financière de l'entrepise est le chiffre d'affaires total")

# Calcul du chiffre d'affaires total
df['Total Price'] = df['Quantity'] * df['Unit Price']
chiffre_affaires_total = df['Total Price'].sum()

# Le chiffre d'affaires total
st.header("Chiffre d'affaires total")
st.write(f"Chiffre d'affaires total: {chiffre_affaires_total}")

# Le chiffre d'affaires par produit
chiffre_affaires_par_produit = df.groupby('Product Name')['Total Price'].sum().sort_values(ascending=False)

# Le chiffre d'affaires par produit
st.header("Chiffre d'affaires par produit")
st.write(chiffre_affaires_par_produit)

# Les transactions par moyen de paiement
moyen_paiement_utilise = df['Payment Method'].value_counts()

# Affichage des transactions par moyen de paiement
st.header("Transactions par moyen de paiement")
st.write(moyen_paiement_utilise)
st.write("Le moyen de payement le plus utilisé est Cash")

# Calcul du chiffre d'affaires par pays
chiffre_affaires_par_pays = df.groupby('Country')['Total Price'].sum().sort_values(ascending=False)

# Affichage du chiffre d'affaires par pays
st.header("Chiffre d'affaires par pays")
st.write(chiffre_affaires_par_pays)
st.write("Les ventes sont plus élevées au Portugal")

# Conversion de la colonne 'Transaction Date' en datetime
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

# Ajout d'une colonne 'Month-Year'
df['Month-Year'] = df['Transaction Date'].dt.to_period('M')

# Calcul du chiffre d'affaires par mois
chiffre_affaires_par_mois = df.groupby('Month-Year')['Total Price'].sum()

# Affichage de la tendance des ventes
st.header("Tendance des ventes mensuelles")
fig, ax = plt.subplots()
chiffre_affaires_par_mois.plot(kind='line', ax=ax)
st.pyplot(fig)

# Préparation des données pour la régression linéaire
df['Month'] = df['Transaction Date'].dt.month
df['Year'] = df['Transaction Date'].dt.year

# Création d'un dataframe pour les ventes mensuelles
df_monthly = df.groupby(['Year', 'Month'])['Total Price'].sum().reset_index()

# Séparation des données en entrainement et test
X = df_monthly[['Year', 'Month']]
y = df_monthly['Total Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédiction des ventes pour mai 2024
prediction = model.predict(np.array([[2024, 5]]))

# Affichage la prédiction
st.header("Prédiction du chiffre d'affaires pour mai 2024")
st.write(f"Prédiction du chiffre d'affaires pour mai 2024: {prediction[0]}")
