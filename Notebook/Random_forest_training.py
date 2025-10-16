# Modello di Random Forest per la previsione dei prezzi delle auto

#Import delle librerie e del dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
df = pd.read_csv(r"../data/processed/database_cleaned_2.csv")

# Variabile Target (obiettivo)
target_variable = 'Price'
y = df[target_variable]
# Rimozione delle colonne non necessarie per la predizione
features = df.drop(columns=['Date', 'Annual Income', target_variable])

# Identificazione delle colonne categoriche
categorical_features = features.select_dtypes(include=['object']).columns

# Pulizia dei valori della colonna 'Engine'
if 'Engine' in categorical_features:
    features['Engine'] = features['Engine'].str.replace('\xa0', ' ', regex=False).str.strip()

# Applicazione del One-Hot Encoding sulle variabili categoriche
X = pd.get_dummies(features, columns=categorical_features, drop_first=True)
print(f"Dimensioni del dataset codificato: {X.shape}")

## Split 80% training e 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

print(f"Dimensioni Training Set (X_train): {X_train.shape}")
print(f"Dimensioni Test Set (X_test): {X_test.shape}")

## Addestramento del Modello Random Forest

# Inizializzazione del modello Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=300, random_state=69, n_jobs=-1)

# Addestramento del modello sul set di training
rf_model.fit(X_train, y_train)
print("\nAddestramento del modello Random Forest completato.")

##  Predizione e Valutazione

y_pred = rf_model.predict(X_test)

# Valutazione delle metriche di regressione
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Risultati della Valutazione del Modello ---")
print(f"Errore Assoluto Medio (MAE): {mae:,.2f}")
print(f"Radice dell'Errore Quadratico Medio (RMSE): {rmse:,.2f}")
print(f"Coefficiente di Determinazione (R-quadro): {r2:.4f}")

# Importanza delle Features
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\n--- Top 10 Importanza delle Features ---")
print(feature_importances.head(10))