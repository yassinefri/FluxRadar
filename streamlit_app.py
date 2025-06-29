import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import networkx as nx
from src.api.load_graph import load_graph
from src.routing.utils import nearest_node

# Configuration de la page
st.set_page_config(
    page_title="FluxRadar Delivery Simulation",
    page_icon="🚚",
    layout="wide"
)

@st.cache_data
def load_stores():
    try:
        df = pd.read_csv("data/processed/monoprix_nodes.csv")
        
        # Renommer les colonnes X/Y vers longitude/latitude
        if 'X' in df.columns and 'Y' in df.columns:
            df = df.rename(columns={'X': 'longitude', 'Y': 'latitude'})
        
        # Vérifier que les colonnes existent maintenant
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            st.error(f"❌ Colonnes manquantes. Colonnes disponibles: {df.columns.tolist()}")
            return pd.DataFrame()
        
        return df
    except FileNotFoundError:
        st.error("❌ Fichier monoprix_nodes.csv non trouvé")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def get_graph():
    try:
        return load_graph("data/processed/graph_paris.graphml")
    except:
        st.error("❌ Impossible de charger le graphe")
        return None

def create_paris_map(stores_df, selected_coords=None, selected_stores=None, optimal_route=None):
    """Crée une carte interactive avec Plotly - affiche UNIQUEMENT les magasins sélectionnés"""
    # Centre sur Paris
    center_lat, center_lon = 48.8566, 2.3522
    
    fig = go.Figure()
    
    # UNIQUEMENT les magasins sélectionnés (pas tous les magasins)
    if selected_stores and len(selected_stores) > 0 and not stores_df.empty:
        if 'latitude' in stores_df.columns and 'longitude' in stores_df.columns:
            # Extraire les adresses des magasins sélectionnés
            selected_addresses = []
            for store in selected_stores:
                # Format: "Nom - Adresse"
                if " - " in store:
                    address = store.split(" - ", 1)[1].strip()  # Prendre tout après le premier " - "
                    selected_addresses.append(address)
            
            # Filtrer le DataFrame pour ne garder QUE les magasins sélectionnés (par adresse)
            filtered_stores = stores_df[stores_df['address'].isin(selected_addresses)]
            
            # Ajouter SEULEMENT les magasins Monoprix sélectionnés
            if not filtered_stores.empty:
                fig.add_trace(go.Scattermap(
                    lat=filtered_stores['latitude'],
                    lon=filtered_stores['longitude'],
                    mode='markers',
                    marker=dict(
                        size=14,
                        color='red',
                        symbol='circle'
                    ),
                    text=filtered_stores['name'] + '<br>' + filtered_stores['address'],
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    name='🏪 Monoprix Sélectionnés'
                ))
    
    # Ajouter le point de départ si disponible
    if selected_coords:
        fig.add_trace(go.Scattermap(
            lat=[selected_coords[0]],
            lon=[selected_coords[1]],
            mode='markers',
            marker=dict(
                size=18,
                color='green',
                symbol='star'
            ),
            text='Point de départ',
            hovertemplate='<b>Point de départ</b><extra></extra>',
            name='🚀 Départ'
        ))
    
    # Ajouter le trajet optimal si disponible
    if optimal_route and 'route_coords' in optimal_route:
        route_coords = optimal_route['route_coords']
        if route_coords:
            lats, lons = zip(*route_coords)
            
            # Ligne du trajet
            fig.add_trace(go.Scattermap(
                lat=lats,
                lon=lons,
                mode='lines',
                line=dict(
                    width=4,
                    color='blue'
                ),
                name='🛣️ Trajet Optimal',
                hovertemplate='Trajet optimal<extra></extra>'
            ))
            
            # Point d'arrivée
            if len(route_coords) > 1:
                fig.add_trace(go.Scattermap(
                    lat=[lats[-1]],
                    lon=[lons[-1]],
                    mode='markers',
                    marker=dict(
                        size=18,
                        color='orange',
                        symbol='diamond'
                    ),
                    text='Point d\'arrivée',
                    hovertemplate='<b>🏁 Arrivée</b><extra></extra>',
                    name='🏁 Arrivée'
                ))
    
    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=11
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True
    )
    
    return fig

def calculate_route_metrics(G, start_coords, end_coords, avg_speed_kmh=25):
    """Calcule les métriques d'une route avec OSMnx et Dijkstra"""
    if G is None:
        return None
    
    try:
        import osmnx as ox
        
        # Trouver les nœuds les plus proches
        orig_node = nearest_node(G, start_coords)
        dest_node = nearest_node(G, end_coords)
        
        # Calculer le chemin le plus court avec Dijkstra (déjà implémenté dans NetworkX)
        path = nx.shortest_path(G, orig_node, dest_node, weight="length")
        
        # Calculer la distance totale
        distance_m = nx.shortest_path_length(G, orig_node, dest_node, weight="length")
        
        # Calculer le temps de trajet avec la vitesse personnalisée
        # Convertir distance en km puis calculer le temps en minutes
        distance_km = distance_m / 1000
        travel_time_hours = distance_km / avg_speed_kmh
        travel_time_minutes = travel_time_hours * 60
        
        # Obtenir les coordonnées du chemin
        route_coords = []
        for node in path:
            node_data = G.nodes[node]
            if 'y' in node_data and 'x' in node_data:  # OSMnx format
                route_coords.append((node_data['y'], node_data['x']))
            elif 'lat' in node_data and 'lon' in node_data:
                route_coords.append((node_data['lat'], node_data['lon']))
        
        return {
            'distance': distance_m,
            'duration': travel_time_minutes,
            'path': path,
            'route_coords': route_coords,
            'nodes_count': len(path)
        }
    except Exception as e:
        st.error(f"Erreur lors du calcul de route: {str(e)}")
        return None

def calculate_optimal_route(G, start_coords, stores_df, selected_stores, avg_speed_kmh=25):
    """Calcule le trajet optimal vers tous les magasins sélectionnés avec Dijkstra"""
    if G is None or not selected_stores:
        return None
    
    try:
        # Obtenir les adresses des magasins sélectionnés (pas les noms car tous = "Monoprix")
        selected_addresses = []
        for store in selected_stores:
            if " - " in store:
                address = store.split(" - ", 1)[1].strip()  # Prendre tout après le premier " - "
                selected_addresses.append(address)
        
        filtered_stores = stores_df[stores_df['address'].isin(selected_addresses)]
        
        if filtered_stores.empty:
            return None
        
        # Calculer les métriques pour chaque magasin avec Dijkstra
        results = []
        for _, store in filtered_stores.iterrows():
            dest_coords = (store['latitude'], store['longitude'])
            metrics = calculate_route_metrics(G, start_coords, dest_coords, avg_speed_kmh)
            if metrics:
                results.append({
                    'store_name': store['name'],
                    'store_address': store['address'],
                    'coords': dest_coords,
                    'distance': metrics['distance'],
                    'duration': metrics['duration'],
                    'path': metrics['path'],
                    'route_coords': metrics['route_coords']
                })
        
        if not results:
            return None
        
        # Trouver le magasin le plus proche (temps minimal)
        optimal_store = min(results, key=lambda x: x['duration'])
        
        return {
            'optimal_store': optimal_store,
            'all_results': results,
            'route_coords': optimal_store['route_coords'],
            'total_time': optimal_store['duration'],
            'total_distance': optimal_store['distance']
        }
    
    except Exception as e:
        st.error(f"Erreur lors du calcul du trajet optimal: {str(e)}")
        return None

def main():
    # Header avec style
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #1f77b4;'>🚚 FluxRadar Delivery Simulation</h1>
        <p style='font-size: 18px; color: #666;'>Optimisez vos livraisons vers les magasins Monoprix</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement des données
    stores_df = load_stores()
    if stores_df.empty:
        st.stop()
    
    G = get_graph()
    
    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("⚙️ Paramètres de Navigation")
        
        # Paramètres pour la vitesse
        st.subheader("� Vitesse de déplacement")
        avg_speed = st.slider("Vitesse moyenne (km/h)", 15, 50, 25, 
                             help="Vitesse moyenne en milieu urbain parisien")
        
        st.header("🏪 Magasins")
        if 'name' in stores_df.columns and 'address' in stores_df.columns:
            display_list = (stores_df["name"] + " - " + stores_df["address"]).tolist()
            selected_stores = st.multiselect("Sélectionnez les Monoprix", display_list)
        else:
            st.error("Colonnes 'name' ou 'address' manquantes")
            selected_stores = []
    
    # Layout principal en colonnes
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Carte interactive")
        
        # État pour stocker les coordonnées sélectionnées
        if 'selected_coords' not in st.session_state:
            st.session_state.selected_coords = None
        
        # Boutons pour sélectionner le point de départ
        st.write("📍 **Sélection du point de départ:**")
        coords_input = st.text_input(
            "Entrez les coordonnées (latitude, longitude)", 
            placeholder="48.8566, 2.3522"
        )
        
        if st.button("📌 Définir le point de départ"):
            try:
                lat, lon = map(float, coords_input.split(','))
                st.session_state.selected_coords = (lat, lon)
                st.success(f"Point de départ défini: {lat:.5f}, {lon:.5f}")
            except:
                st.error("Format invalide. Utilisez: latitude, longitude")
        
        # Affichage de la carte
        # État pour stocker le trajet optimal
        if 'optimal_route' not in st.session_state:
            st.session_state.optimal_route = None
        
        # Bouton pour calculer le trajet optimal
        if st.session_state.selected_coords and selected_stores and G:
            if st.button("🚀 Calculer le trajet optimal", type="primary"):
                with st.spinner("Calcul du trajet optimal avec Dijkstra..."):
                    optimal_route = calculate_optimal_route(
                        G, st.session_state.selected_coords, 
                        stores_df, selected_stores, avg_speed
                    )
                    st.session_state.optimal_route = optimal_route
                    
                    if optimal_route:
                        st.success(f"🎯 Trajet optimal calculé vers {optimal_route['optimal_store']['store_address']}")
                        st.info(f"⏱️ Temps estimé: {optimal_route['total_time']:.1f} min | 📏 Distance: {optimal_route['total_distance']:.0f}m | 🚗 Vitesse: {avg_speed} km/h")
                    else:
                        st.error("❌ Impossible de calculer le trajet optimal")
        
        # Affichage de la carte avec le trajet optimal
        fig = create_paris_map(stores_df, st.session_state.selected_coords, 
                              selected_stores, st.session_state.optimal_route)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Informations")
        
        # Statistiques des magasins
        st.metric("🏪 Nombre de Monoprix", len(stores_df))
        st.metric("📍 Magasins sélectionnés", len(selected_stores))
        
        # Informations sur le trajet optimal
        if st.session_state.optimal_route:
            optimal = st.session_state.optimal_route
            st.success("✅ Trajet optimal calculé")
            st.metric("🎯 Destination", optimal['optimal_store']['store_name'])
            st.metric("⏱️ Temps optimal", f"{optimal['total_time']:.1f} min")
            st.metric("📏 Distance", f"{optimal['total_distance']:.0f} m")
            
            if st.button("🗑️ Effacer le trajet"):
                st.session_state.optimal_route = None
                st.rerun()
        else:
            if st.session_state.selected_coords:
                st.success("✅ Point de départ défini")
            else:
                st.info("ℹ️ Définissez un point de départ")
    
    # Simulation et résultats détaillés
    if st.session_state.optimal_route:
        st.subheader("🎯 Simulation du Trajet Optimal")
        
        optimal = st.session_state.optimal_route
        
        # Informations détaillées
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### 📍 Destination Optimale")
            st.write(f"**🏪 Magasin:** {optimal['optimal_store']['store_name']}")
            st.write(f"**📍 Adresse:** {optimal['optimal_store']['store_address']}")
            st.write(f"**⏱️ Temps de trajet:** {optimal['total_time']:.1f} minutes")
            st.write(f"**📏 Distance:** {optimal['total_distance']:.0f} mètres")
        
        with col4:
            st.markdown("### 📊 Comparaison avec les autres magasins")
            if len(optimal['all_results']) > 1:
                comparison_data = []
                for result in optimal['all_results']:
                    is_optimal = result['store_name'] == optimal['optimal_store']['store_name']
                    comparison_data.append({
                        'Magasin': result['store_name'],
                        'Temps (min)': f"{result['duration']:.1f}",
                        'Distance (m)': f"{result['distance']:.0f}",
                        'Optimal': "🎯" if is_optimal else ""
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
            else:
                st.info("Un seul magasin sélectionné")
        
        # Graphique de comparaison
        if len(optimal['all_results']) > 1:
            fig_comparison = px.bar(
                x=[r['store_name'] for r in optimal['all_results']],
                y=[r['duration'] for r in optimal['all_results']],
                title="⏱️ Comparaison des temps de trajet",
                labels={'x': 'Magasins', 'y': 'Temps (minutes)'},
                color=[r['duration'] for r in optimal['all_results']],
                color_continuous_scale='RdYlGn_r'
            )
            
            # Marquer le magasin optimal
            optimal_idx = next(i for i, r in enumerate(optimal['all_results']) 
                             if r['store_name'] == optimal['optimal_store']['store_name'])
            fig_comparison.add_annotation(
                x=optimal_idx,
                y=optimal['total_time'],
                text="🎯 OPTIMAL",
                showarrow=True,
                arrowhead=2,
                arrowcolor="green",
                font=dict(color="green", size=12)
            )
            
            fig_comparison.update_layout(height=400)
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Instructions pour l'utilisateur
    # Instructions pour l'utilisateur
    if not st.session_state.selected_coords and not selected_stores:
        st.info("📋 **Instructions:**\n"
                "1. 📍 Définissez un point de départ\n"
                "2. 🏪 Sélectionnez un ou plusieurs magasins Monoprix\n"
                "3. 🚀 Cliquez sur 'Calculer le trajet optimal'")
    elif not st.session_state.selected_coords:
        st.info("📍 Définissez un point de départ pour calculer les routes")
    elif not selected_stores:
        st.info("🏪 Sélectionnez au moins un magasin Monoprix dans la barre latérale")
    elif not st.session_state.optimal_route:
        st.info("🚀 Cliquez sur 'Calculer le trajet optimal' pour voir la simulation")

if __name__ == "__main__":
    main()