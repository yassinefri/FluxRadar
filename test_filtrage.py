"""
Test du filtrage par adresse
"""
import pandas as pd

# Test de chargement des données
print("Test de filtrage par adresse...")
df = pd.read_csv("data/processed/monoprix_nodes.csv")

# Renommer les colonnes
if 'X' in df.columns and 'Y' in df.columns:
    df = df.rename(columns={'X': 'longitude', 'Y': 'latitude'})

# Test de filtrage par adresse
selected_stores = ["Monoprix - 118/130 Avenue Jean Jaurès", "Monoprix - 49 Rue d'Auteuil"]
selected_addresses = []
for store in selected_stores:
    if " - " in store:
        address = store.split(" - ", 1)[1].strip()
        selected_addresses.append(address)

print(f"Magasins sélectionnés: {selected_stores}")
print(f"Adresses extraites: {selected_addresses}")

filtered_stores = df[df['address'].isin(selected_addresses)]

print(f"Magasins filtrés: {len(filtered_stores)}")
print("Magasins filtrés:")
for _, store in filtered_stores.iterrows():
    print(f"  - {store['name']} à {store['address']}")

if len(filtered_stores) == len(selected_stores):
    print("SUCCÈS ! Le filtrage par adresse fonctionne correctement.")
else:
    print(" ÉCHEC ! Le filtrage ne fonctionne pas correctement.")
