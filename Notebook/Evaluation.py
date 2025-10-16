import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pickle 

MODEL_PATH = '../data/results/best_random_forest_model.pkl'
X_TEST_PATH = '../data/results/X_test_for_evaluation.csv'
Y_TEST_PATH = '../data/results/y_test_for_evaluation.csv'

print("Inizio della valutazione del modello...")
try:
    # Carica il modello addestrato
    with open(MODEL_PATH, 'rb') as file:
        loaded_model = pickle.load(file)
    print("Modello Random Forest caricato con successo.")

    # Carica i set di test
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze() 
    print("Dati di test caricati con successo.")

except FileNotFoundError:
    print(f"ERRORE: I file necessari non sono stati trovati. Assicurati che l'esecuzione di training sia completata.")
    exit()

print("\nAvvio della predizione sul set di test...")
y_pred = loaded_model.predict(X_test)

# Valutazione delle metriche
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = np.mean(np.abs(y_test - y_pred)) # Aggiungo MAE per completezza

print("\n--- Risultati Finali del Modello Ottimizzato (Valutazione) ---")
print(f"Radice dell'Errore Quadratico Medio (RMSE): {rmse:,.2f}")
print(f"Errore Assoluto Medio (MAE): {mae:,.2f}")
print(f"Coefficiente di Determinazione (R-quadro): {r2:.4f}")

print("\nValutazione completata.")