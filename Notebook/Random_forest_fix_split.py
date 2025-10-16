import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import numpy as np
import pickle
df_main = pd.read_csv(r"../data/processed/database_cleaned_2.csv")
df_split = pd.read_csv(r"../data/processed/split_v1.csv")

# Assegnazione dell'indice al DataFrame principale per unire con lo split
df_main.reset_index(inplace=True)
df_main.rename(columns={'index': 'row_index'}, inplace=True)

# Unione dei due DataFrames sulla colonna 'row_index'
df_merged = pd.merge(df_main, df_split, on='row_index', how='left')

# Definizione di Target e Features
target_variable = 'Price'
y = df_merged[target_variable]
features = df_merged.drop(columns=['Date', target_variable, 'row_index', 'split', 'Annual Income'])

# Identificazione delle colonne categoriche e pulizia dei valori
categorical_features = features.select_dtypes(include=['object']).columns

if 'Engine' in categorical_features:
    features['Engine'] = features['Engine'].str.replace('\xa0', ' ', regex=False).str.strip()

# Applicazione del One-Hot Encoding sulle variabili categoriche
X = pd.get_dummies(features, columns=categorical_features, drop_first=True)

# Suddivisione usando la colonna 'split' dal file esterno
X_train = X[df_merged['split'] == 'train']
X_test = X[df_merged['split'] == 'test']
y_train = y[df_merged['split'] == 'train']
y_test = y[df_merged['split'] == 'test']

print(f"Dimensioni Training Set (X_train): {X_train.shape}")
print(f"Dimensioni Test Set (X_test): {X_test.shape}")

# Definizione della griglia di parametri da testare
param_dist = {'n_estimators': sp_randint(200, 1000), 'max_depth': [10, 15, 20, 30, None],'min_samples_split': sp_randint(2, 11), 'min_samples_leaf': sp_randint(1, 11) }

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(estimator=rf_base, param_distributions=param_dist, n_iter=20, cv=3, scoring='r2', random_state=42, n_jobs=-1, verbose=2)

print("\nAvvio della ricerca sugli iperparametri.")
random_search.fit(X_train, y_train)
best_rf_model = random_search.best_estimator_

print("\n--- Risultati del Tuning Light ---")
print(f"Migliori Parametri Trovati: {random_search.best_params_}")
print(f"Miglior R-quadro (su Cross-Validation): {random_search.best_score_:.4f}")

# 4. SALVATAGGIO DEI RISULTATI
# 4. SALVATAGGIO DEI RISULTATI NELLA CARTELLA data/processed 💾
MODEL_PATH = '../data/processed/best_random_forest_model.pkl'
X_TEST_PATH = '../data/processed/X_test_for_evaluation.csv'
Y_TEST_PATH = '../data/processed/y_test_for_evaluation.csv'

try:
    # Salva il miglior modello
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(best_rf_model, file)
    print(f"\nModello Random Forest salvato in '{MODEL_PATH}'")

    # Salva i dati di test
    X_test.to_csv(X_TEST_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False, header=True)
    print(f"Dati di test salvati in '{X_TEST_PATH}' e '{Y_TEST_PATH}'")
    
except Exception as e: 
    print(f"Errore durante il salvataggio dei file: {e}")

print("\nAddestramento e Salvataggio completati.")