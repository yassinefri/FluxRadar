try:
    import osmnx as ox
except ImportError:
    import networkx as nx
    import xml.etree.ElementTree as ET
    print("⚠️ OSMnx non disponible, utilisation de NetworkX basique")
    ox = None
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

    if ox is not None:
        # Utilisation d'OSMnx si disponible
        graph = ox.load_graphml(filepath)
    else:
        # Fallback avec NetworkX
        import networkx as nx
        graph = nx.read_graphml(filepath)
    
    print("Graphe chargé avec succès.")
    return graph


# Exemple d’utilisation :
if __name__ == "__main__":
    G = load_graph("data/processed/graph_paris.graphml")
