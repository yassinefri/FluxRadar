"""
Utilitaires pour le routage et les calculs de graphes
"""
import networkx as nx
import numpy as np


def nearest_node(G, coords):
    """
    Trouve le nœud le plus proche dans le graphe pour des coordonnées données
    
    Args:
        G: Graphe NetworkX
        coords: Tuple (latitude, longitude)
    
    Returns:
        Node ID du nœud le plus proche
    """
    lat, lon = coords
    
    # Récupérer les coordonnées de tous les nœuds
    nodes_coords = []
    node_ids = []
    
    for node_id, data in G.nodes(data=True):
        if 'y' in data and 'x' in data:  # OSMnx utilise 'y' pour lat et 'x' pour lon
            nodes_coords.append((data['y'], data['x']))
            node_ids.append(node_id)
        elif 'lat' in data and 'lon' in data:
            nodes_coords.append((data['lat'], data['lon']))
            node_ids.append(node_id)
    
    if not nodes_coords:
        raise ValueError("Aucune coordonnée trouvée dans les nœuds du graphe")
    
    # Calculer les distances
    nodes_coords = np.array(nodes_coords)
    target = np.array([lat, lon])
    
    # Distance euclidienne simple (pour de petites distances, c'est suffisant)
    distances = np.sqrt(np.sum((nodes_coords - target) ** 2, axis=1))
    
    # Retourner l'ID du nœud le plus proche
    nearest_idx = np.argmin(distances)
    return node_ids[nearest_idx]


def calculate_route_distance(G, path):
    """
    Calcule la distance totale d'un chemin dans le graphe
    
    Args:
        G: Graphe NetworkX
        path: Liste des nœuds du chemin
    
    Returns:
        Distance totale en mètres
    """
    if len(path) < 2:
        return 0
    
    total_distance = 0
    for i in range(len(path) - 1):
        edge_data = G.get_edge_data(path[i], path[i + 1])
        if edge_data and 'length' in edge_data:
            total_distance += edge_data['length']
    
    return total_distance


def get_route_coordinates(G, path):
    """
    Récupère les coordonnées des nœuds d'un chemin
    
    Args:
        G: Graphe NetworkX
        path: Liste des nœuds du chemin
    
    Returns:
        Liste de tuples (latitude, longitude)
    """
    coordinates = []
    
    for node_id in path:
        data = G.nodes[node_id]
        if 'y' in data and 'x' in data:  # OSMnx format
            coordinates.append((data['y'], data['x']))
        elif 'lat' in data and 'lon' in data:
            coordinates.append((data['lat'], data['lon']))
    
    return coordinates
