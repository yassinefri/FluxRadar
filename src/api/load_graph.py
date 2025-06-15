import osmnx as ox
import os


def load_graph(filepath: str):
    """
    Charge un graphe OSMNX sauvegardé en .graphml.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier non trouvé : {filepath}")

    graph = ox.load_graphml(filepath)
    print("Graphe chargé avec succès.")
    return graph


# Exemple d'utilisation :
# G = load_graph("data/processed/graph_paris_lion.graphml")
