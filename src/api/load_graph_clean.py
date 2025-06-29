try:
    import osmnx as ox
except ImportError:
    import networkx as nx
    import xml.etree.ElementTree as ET
    print("OSMnx not available, using basic NetworkX")
    ox = None
import os


def load_graph(filepath: str):
    """
    Load a saved OSMnx graph from .graphml file.

    Args:
        filepath (str): Path to the .graphml file

    Returns:
        networkx.MultiDiGraph: The loaded road graph
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if ox is not None:
        # Use OSMnx if available
        graph = ox.load_graphml(filepath)
    else:
        # Fallback with NetworkX
        import networkx as nx
        graph = nx.read_graphml(filepath)
    
    print("Graph loaded successfully.")
    return graph


# Usage example:
if __name__ == "__main__":
    G = load_graph("data/processed/graph_paris.graphml")
