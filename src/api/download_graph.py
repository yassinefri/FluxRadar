import osmnx as ox 

lieu = "Paris, France"  


G = ox.graph_from_place(lieu, network_type="drive")

ox.plot_graph(G)

ox.save_graphml(G, filepath="/data/graph_paris.graphml")

print("saved successfully to ../data/graph_paris.graphml")
