import os
import osmnx as ox
import networkx as nx
import folium
from src.api.load_graph import load_graph

# 1. Charger le graphe sauvegardé
graph_path = "data/processed/graph_paris.graphml"
G = load_graph(graph_path)

# 2. Définir les coordonnées GPS (latitude, longitude)
origin_point = (48.844, 2.373)  # Exemple : Gare de Lyon
destination_point = (48.860, 2.335)  # Exemple : Le Louvre

# 3. Trouver les nœuds les plus proches dans le graphe
origin_node = ox.distance.nearest_nodes(G, X=origin_point[1], Y=origin_point[0])
destination_node = ox.distance.nearest_nodes(
    G, X=destination_point[1], Y=destination_point[0]
)

# 4. Calculer le plus court chemin (en mètres) avec Dijkstra
shortest_path = nx.shortest_path(G, origin_node, destination_node, weight="length")

# 5. Extraire les coordonnées du trajet
route_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in shortest_path]

# 6. Créer une carte centrée sur le point de départ
m = folium.Map(location=origin_point, zoom_start=14)

# 7. Ajouter le tracé de l'itinéraire
folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.8).add_to(m)

# 8. Ajouter un marqueur pour le départ
folium.Marker(
    location=origin_point, popup="Départ", icon=folium.Icon(color="green")
).add_to(m)

# 9. Ajouter un marqueur pour l’arrivée
folium.Marker(
    location=destination_point, popup="Arrivée", icon=folium.Icon(color="red")
).add_to(m)

# 10. Créer le dossier s'il n’existe pas
os.makedirs("outputs", exist_ok=True)

# 11. Sauvegarder la carte dans un fichier HTML
m.save("outputs/itineraire_paris.html")
print("✅ Itinéraire généré et carte enregistrée dans outputs/itineraire_paris.html")
