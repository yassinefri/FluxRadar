import osmnx as ox
import os


def load_graph(filepath: str):
    """
    Charge un graphe OSMNX sauvegardé en .graphml.

    Args:
        filepath (str): Chemin du fichier .graphml

    Returns:
        networkx.MultiDiGraph: Le graphe routier chargé
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier pas trouvé : {filepath}")

    graph = ox.load_graphml(filepath)
    print("Graphe chargé avec succès.")
    return graph


# Exemple d’utilisation :
if __name__ == "__main__":
    G = load_graph("data/processed/graph_paris.graphml")
