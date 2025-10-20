import pandas as pd
import pickle
from sklearn.tree import export_graphviz
import graphviz
import os
MODEL_PATH = '../data/results/best_random_forest_model.pkl'
RESULTS_DIR = os.path.dirname(MODEL_PATH)
GRAPH_DIR = os.path.join(RESULTS_DIR, 'model_visualizations') 
os.makedirs(GRAPH_DIR, exist_ok=True)


try:    # 1. CARICAMENTO DEL MODELLO RANDOM FOREST
    with open(MODEL_PATH, 'rb') as file:
        loaded_model = pickle.load(file)
    print("Modello Random Forest caricato con successo.")

    # 2. CARICAMENTO DELLE FEATURE NAMES
    X_TEST_PATH = '../data/results/X_test_for_evaluation.csv'
    X_test = pd.read_csv(X_TEST_PATH)
    feature_names = X_test.columns.tolist()

    # 3. SELEZIONE E VISUALIZZAZIONE DI UN SINGOLO ALBERO
    estimator = loaded_model.estimators_[0] 
    
    dot_file_path = os.path.join(GRAPH_DIR, 'tree_0.dot')
    export_graphviz(estimator, out_file=dot_file_path, feature_names=feature_names,max_depth=4,label='none', node_ids=True, rounded=True, precision=1, filled=True, proportion=False,special_characters=True)
    print(f"\nAlbero decisionale esportato in formato DOT: '{dot_file_path}'")
    # 4. CONVERSIONE DEL FILE DOT IN PNG
    try:
        with open(dot_file_path) as f:
            dot_graph = f.read()
        
        png_file_path = os.path.join(GRAPH_DIR, 'tree_0.png')
        graph = graphviz.Source(dot_graph)
        graph.render(png_file_path, view=False, format='png', cleanup=True) 
        print(f"Albero decisionale convertito in PNG: '{png_file_path}'")
        print("\nCOMPLETATO: Controlla la cartella 'model_visualizations' per i file DOT e PNG.")
        
    except graphviz.backend.ExecutableNotFound:
         print("\nATTENZIONE: Impossibile convertire in PNG.")
         print("Per la conversione, installa il programma eseguibile Graphviz sul sistema e aggiungilo al PATH.")
         print("Puoi comunque aprire il file .dot con strumenti online per visualizzarlo.")
    except Exception as e:
         print(f"ATTENZIONE: Errore durante la conversione PNG/DOT. Verifica i token. Errore: {e}")
except FileNotFoundError:
    print(f"ERRORE: File non trovato. Assicurati che i file del modello e del test siano in '{os.path.abspath('../data/results/')}'")
except Exception as e:
    print(f"Si è verificato un errore critico durante il caricamento o la preparazione: {e}")