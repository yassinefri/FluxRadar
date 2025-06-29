"""
Test rapide pour vérifier le filtrage des magasins
"""
import pandas as pd

def test_store_filtering():
    """Test le filtrage des magasins sélectionnés"""
    print("🧪 Test du filtrage des magasins")
    
    # Charger les données
    df = pd.read_csv("data/processed/monoprix_nodes.csv")
    if 'X' in df.columns and 'Y' in df.columns:
        df = df.rename(columns={'X': 'longitude', 'Y': 'latitude'})
    
    print(f"📊 Total magasins: {len(df)}")
    print(f"📋 Premiers magasins: {df['name'].head().tolist()}")
    
    # Simuler une sélection
    selected_stores = ["Monoprix - 118/130 Avenue Jean Jaurès", "Monoprix - 49 Rue d'Auteuil"]
    selected_stores_names = [store.split(" - ")[0].strip() for store in selected_stores]
    
    print(f"🎯 Magasins sélectionnés: {selected_stores}")
    print(f"🔍 Noms extraits: {selected_stores_names}")
    
    # Filtrer
    filtered_stores = df[df['name'].isin(selected_stores_names)]
    print(f"✅ Magasins filtrés: {len(filtered_stores)}")
    print(f"📍 Magasins trouvés: {filtered_stores['name'].tolist()}")
    
    if len(filtered_stores) == len(selected_stores_names):
        print("✅ Filtrage réussi !")
    else:
        print("❌ Problème de filtrage")
        print("Vérifiez que les noms correspondent exactement")

if __name__ == "__main__":
    test_store_filtering()
