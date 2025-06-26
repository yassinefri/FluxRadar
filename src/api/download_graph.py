import osmnx as ox

def download_and_save_graph(place_name: str, filepath: str):
    """
    Télécharge le graphe routier d'un lieu et le sauvegarde au format .graphml

    Args:
        place_name (str): Nom du lieu à télécharger (ville, quartier, etc.)
        filepath (str): Chemin du fichier où sauvegarder le graphe
    """
    print(f"Téléchargement du graphe pour : {place_name}")
    G = ox.graph_from_place(place_name, network_type="drive")
    
    print("Affichage du graphe...")
    ox.plot_graph(G)
    
    print(f"Sauvegarde du graphe vers : {filepath}")
    ox.save_graphml(G, filepath=filepath)
    print("✅ Graphe téléchargé et sauvegardé avec succès.")

# Exemple d’utilisation
if __name__ == "__main__":
    download_and_save_graph(
        place_name="Paris, France",
        filepath="data/processed/graph_paris.graphml"
    )