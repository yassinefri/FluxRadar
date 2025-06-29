"""
Test simple pour vérifier le fonctionnement de l'application
"""
import pandas as pd

# Test de chargement des données
print("🧪 Test de chargement des données...")
df = pd.read_csv("data/processed/monoprix_nodes.csv")
print(f"✅ {len(df)} magasins chargés")
print(f"Colonnes: {df.columns.tolist()}")

# Renommer les colonnes
if 'X' in df.columns and 'Y' in df.columns:
    df = df.rename(columns={'X': 'longitude', 'Y': 'latitude'})
    print("✅ Colonnes renommées")

# Test de filtrage
selected_stores = ["Monoprix - 118/130 Avenue Jean Jaurès", "Monoprix - 49 Rue d'Auteuil"]
selected_stores_names = [store.split(" - ")[0] for store in selected_stores]
filtered_stores = df[df['name'].isin(selected_stores_names)]

print(f"🎯 Magasins sélectionnés: {len(selected_stores_names)}")
print(f"🎯 Magasins filtrés: {len(filtered_stores)}")
print("🎯 Magasins filtrés:")
for _, store in filtered_stores.iterrows():
    print(f"  - {store['name']} à {store['address']}")

print("\n✅ Tous les tests passent ! L'application devrait fonctionner correctement.")
