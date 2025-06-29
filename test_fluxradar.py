"""
Script de test pour vérifier le fonctionnement de FluxRadar
"""
import pandas as pd
import sys
import os

def test_data_loading():
    """Test le chargement des données"""
    print("🔍 Test du chargement des données...")
    
    # Test du fichier CSV
    try:
        df = pd.read_csv("data/processed/monoprix_nodes.csv")
        print(f"✅ CSV chargé: {len(df)} magasins trouvés")
        print(f"📊 Colonnes: {df.columns.tolist()}")
        
        # Renommer les colonnes
        if 'X' in df.columns and 'Y' in df.columns:
            df = df.rename(columns={'X': 'longitude', 'Y': 'latitude'})
            print("✅ Colonnes renommées avec succès")
            print(f"📍 Premier magasin: {df.iloc[0]['name']} à ({df.iloc[0]['latitude']:.4f}, {df.iloc[0]['longitude']:.4f})")
        
        return True
    except Exception as e:
        print(f"Erreur CSV: {e}")
        return False

def test_graph_loading():
    """Test le chargement du graphe"""
    print("\n🗺️ Test du chargement du graphe...")
    
    try:
        from src.api.load_graph import load_graph
        G = load_graph("data/processed/graph_paris.graphml")
        print(f"✅ Graphe chargé: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
        return True
    except Exception as e:
        print(f"Erreur graphe: {e}")
        return False

def test_model_loading():
    """Test le chargement du modèle"""
    print("\n🤖 Test du chargement du modèle...")
    
    try:
        import torch
        model = torch.load("src/models/linear_regression_model.pt")
        print("Modèle PyTorch chargé avec succès")
        return True
    except Exception as e:
        print(f"Erreur modèle: {e}")
        return False

def test_routing():
    """Test des fonctions de routage"""
    print("\n🛣️ Test des fonctions de routage...")
    
    try:
        from src.routing.utils import nearest_node
        from src.api.load_graph import load_graph
        
        G = load_graph("data/processed/graph_paris.graphml")
        # Test avec des coordonnées de Paris
        test_coords = (48.8566, 2.3522)  # Notre-Dame
        node = nearest_node(G, test_coords)
        print(f"Nœud le plus proche trouvé: {node}")
        return True
    except Exception as e:
        print(f"Erreur routage: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🚚 FluxRadar - Tests de fonctionnement\n")
    
    tests = [
        test_data_loading,
        test_graph_loading,
        test_model_loading,
        test_routing
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Erreur inattendue: {e}")
            results.append(False)
    
    print(f"\n📋 Résumé des tests:")
    print(f"Tests réussis: {sum(results)}/{len(results)}")
    
    if all(results):
        print("Tous les tests passent ! FluxRadar est opérationnel.")
    else:
        print( "Certains tests échouent. Vérifiez les erreurs ci-dessus.")

if __name__ == "__main__":
    main()
