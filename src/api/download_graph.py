import osmnx as ox

def download_and_save_graph(place_name: str, filepath: str):
    """
    Download road graph of a place and save it as .graphml format

    Args:
        place_name (str): Name of the place to download (city, district, etc.)
        filepath (str): File path where to save the graph
    """
    print(f"Downloading graph for: {place_name}")
    G = ox.graph_from_place(place_name, network_type="drive")
    
    print("Displaying graph...")
    ox.plot_graph(G)
    
    print(f"Saving graph to: {filepath}")
    ox.save_graphml(G, filepath=filepath)
    print("Graph downloaded and saved successfully.")

# Exemple d’utilisation
if __name__ == "__main__":
    download_and_save_graph(
        place_name="Paris, France",
        filepath="data/processed/graph_paris.graphml"
    )