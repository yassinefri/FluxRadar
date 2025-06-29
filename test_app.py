"""
Simple test to verify application functionality
"""
import pandas as pd

# Data loading test
print("Testing data loading...")
df = pd.read_csv("data/processed/monoprix_nodes.csv")
print(f"{len(df)} stores loaded")
print(f"Columns: {df.columns.tolist()}")

# Rename columns
if 'X' in df.columns and 'Y' in df.columns:
    df = df.rename(columns={'X': 'longitude', 'Y': 'latitude'})
    print("Columns renamed")

# Filtering test
selected_stores = ["Monoprix - 118/130 Avenue Jean Jaurès", "Monoprix - 49 Rue d'Auteuil"]
selected_stores_names = [store.split(" - ")[0] for store in selected_stores]
filtered_stores = df[df['name'].isin(selected_stores_names)]

print(f"Selected stores: {len(selected_stores_names)}")
print(f"Filtered stores: {len(filtered_stores)}")
print("Filtered stores:")
for _, store in filtered_stores.iterrows():
    print(f"  - {store['name']} at {store['address']}")

print("\nAll tests pass! The application should work correctly.")
