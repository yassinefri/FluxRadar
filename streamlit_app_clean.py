import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import networkx as nx
from src.api.load_graph import load_graph
from src.routing.utils import nearest_node

# Page configuration
st.set_page_config(
    page_title="FluxRadar Delivery Simulation",
    page_icon="🚚",
    layout="wide"
)

@st.cache_data
def load_stores():
    try:
        df = pd.read_csv("data/processed/monoprix_nodes.csv")
        
        # Rename X/Y columns to longitude/latitude
        if 'X' in df.columns and 'Y' in df.columns:
            df = df.rename(columns={'X': 'longitude', 'Y': 'latitude'})
        
        # Check that columns exist now
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            st.error(f"Missing columns. Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        return df
    except FileNotFoundError:
        st.error("File monoprix_nodes.csv not found")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def get_graph():
    try:
        return load_graph("data/processed/graph_paris.graphml")
    except:
        st.error("Unable to load graph")
        return None

def create_paris_map(stores_df, selected_coords=None, selected_stores=None, optimal_route=None):
    """Creates an interactive map with Plotly - displays ONLY selected stores"""
    # Center on Paris
    center_lat, center_lon = 48.8566, 2.3522
    
    fig = go.Figure()
    
    # ONLY selected stores (not all stores)
    if selected_stores and len(selected_stores) > 0 and not stores_df.empty:
        if 'latitude' in stores_df.columns and 'longitude' in stores_df.columns:
            # Extract addresses of selected stores
            selected_addresses = []
            for store in selected_stores:
                # Format: "Name - Address"
                if " - " in store:
                    address = store.split(" - ", 1)[1].strip()  # Take everything after first " - "
                    selected_addresses.append(address)
            
            # Filter DataFrame to keep ONLY selected stores (by address)
            filtered_stores = stores_df[stores_df['address'].isin(selected_addresses)]
            
            # Add ONLY selected Monoprix stores
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
                    name='Selected Monoprix'
                ))
    
    # Add starting point if available
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
            text='Starting point',
            hovertemplate='<b>Starting point</b><extra></extra>',
            name='Start'
        ))
    
    # Add optimal route if available
    if optimal_route and 'route_coords' in optimal_route:
        route_coords = optimal_route['route_coords']
        if route_coords:
            lats, lons = zip(*route_coords)
            
            # Route line
            fig.add_trace(go.Scattermap(
                lat=lats,
                lon=lons,
                mode='lines',
                line=dict(
                    width=4,
                    color='blue'
                ),
                name='Optimal Route',
                hovertemplate='Optimal route<extra></extra>'
            ))
            
            # Destination point
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
                    text='Destination',
                    hovertemplate='<b>Destination</b><extra></extra>',
                    name='Destination'
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
    """Calculate route metrics with OSMnx and Dijkstra"""
    if G is None:
        return None
    
    try:
        import osmnx as ox
        
        # Find nearest nodes
        orig_node = nearest_node(G, start_coords)
        dest_node = nearest_node(G, end_coords)
        
        # Calculate shortest path with Dijkstra (already implemented in NetworkX)
        path = nx.shortest_path(G, orig_node, dest_node, weight="length")
        
        # Calculate total distance
        distance_m = nx.shortest_path_length(G, orig_node, dest_node, weight="length")
        
        # Calculate travel time with custom speed
        # Convert distance to km then calculate time in minutes
        distance_km = distance_m / 1000
        travel_time_hours = distance_km / avg_speed_kmh
        travel_time_minutes = travel_time_hours * 60
        
        # Get path coordinates
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
        st.error(f"Error calculating route: {str(e)}")
        return None

def calculate_optimal_route(G, start_coords, stores_df, selected_stores, avg_speed_kmh=25):
    """Calculate optimal route to all selected stores with Dijkstra"""
    if G is None or not selected_stores:
        return None
    
    try:
        # Get addresses of selected stores (not names since all = "Monoprix")
        selected_addresses = []
        for store in selected_stores:
            if " - " in store:
                address = store.split(" - ", 1)[1].strip()  # Take everything after first " - "
                selected_addresses.append(address)
        
        filtered_stores = stores_df[stores_df['address'].isin(selected_addresses)]
        
        if filtered_stores.empty:
            return None
        
        # Calculate metrics for each store with Dijkstra
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
        
        # Find closest store (minimal time)
        optimal_store = min(results, key=lambda x: x['duration'])
        
        return {
            'optimal_store': optimal_store,
            'all_results': results,
            'route_coords': optimal_store['route_coords'],
            'total_time': optimal_store['duration'],
            'total_distance': optimal_store['distance']
        }
    
    except Exception as e:
        st.error(f"Error calculating optimal route: {str(e)}")
        return None

def main():
    # Header with style
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #1f77b4;'>FluxRadar Delivery Simulation</h1>
        <p style='font-size: 18px; color: #666;'>Optimize your deliveries to Monoprix stores</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    stores_df = load_stores()
    if stores_df.empty:
        st.stop()
    
    G = get_graph()
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Navigation Settings")
        
        # Speed parameters
        st.subheader("Travel Speed")
        avg_speed = st.slider("Average speed (km/h)", 15, 50, 25, 
                             help="Average speed in Parisian urban environment")
        
        st.header("Stores")
        if 'name' in stores_df.columns and 'address' in stores_df.columns:
            display_list = (stores_df["name"] + " - " + stores_df["address"]).tolist()
            selected_stores = st.multiselect("Select Monoprix stores", display_list)
        else:
            st.error("Missing 'name' or 'address' columns")
            selected_stores = []
    
    # Main layout in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Interactive Map")
        
        # State to store selected coordinates
        if 'selected_coords' not in st.session_state:
            st.session_state.selected_coords = None
        
        # Buttons to select starting point
        st.write("**Starting point selection:**")
        coords_input = st.text_input(
            "Enter coordinates (latitude, longitude)", 
            placeholder="48.8566, 2.3522"
        )
        
        if st.button("Set starting point"):
            try:
                lat, lon = map(float, coords_input.split(','))
                st.session_state.selected_coords = (lat, lon)
                st.success(f"Starting point set: {lat:.5f}, {lon:.5f}")
            except:
                st.error("Invalid format. Use: latitude, longitude")
        
        # Map display
        # State to store optimal route
        if 'optimal_route' not in st.session_state:
            st.session_state.optimal_route = None
        
        # Button to calculate optimal route
        if st.session_state.selected_coords and selected_stores and G:
            if st.button("Calculate optimal route", type="primary"):
                with st.spinner("Calculating optimal route with Dijkstra..."):
                    optimal_route = calculate_optimal_route(
                        G, st.session_state.selected_coords, 
                        stores_df, selected_stores, avg_speed
                    )
                    st.session_state.optimal_route = optimal_route
                    
                    if optimal_route:
                        st.success(f"Optimal route calculated to {optimal_route['optimal_store']['store_address']}")
                        st.info(f"Estimated time: {optimal_route['total_time']:.1f} min | Distance: {optimal_route['total_distance']:.0f}m | Speed: {avg_speed} km/h")
                    else:
                        st.error("Unable to calculate optimal route")
        
        # Display map with optimal route
        fig = create_paris_map(stores_df, st.session_state.selected_coords, 
                              selected_stores, st.session_state.optimal_route)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Information")
        
        # Store statistics
        st.metric("Number of Monoprix", len(stores_df))
        st.metric("Selected stores", len(selected_stores))
        
        # Optimal route information
        if st.session_state.optimal_route:
            optimal = st.session_state.optimal_route
            st.success("Optimal route calculated")
            st.metric("Destination", optimal['optimal_store']['store_name'])
            st.metric("Optimal time", f"{optimal['total_time']:.1f} min")
            st.metric("Distance", f"{optimal['total_distance']:.0f} m")
            
            if st.button("Clear route"):
                st.session_state.optimal_route = None
                st.rerun()
        else:
            if st.session_state.selected_coords:
                st.success("Starting point defined")
            else:
                st.info("Define a starting point")
    
    # Simulation and detailed results
    if st.session_state.optimal_route:
        st.subheader("Optimal Route Simulation")
        
        optimal = st.session_state.optimal_route
        
        # Detailed information
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### Optimal Destination")
            st.write(f"**Store:** {optimal['optimal_store']['store_name']}")
            st.write(f"**Address:** {optimal['optimal_store']['store_address']}")
            st.write(f"**Travel time:** {optimal['total_time']:.1f} minutes")
            st.write(f"**Distance:** {optimal['total_distance']:.0f} meters")
        
        with col4:
            st.markdown("### Comparison with other stores")
            if len(optimal['all_results']) > 1:
                comparison_data = []
                for result in optimal['all_results']:
                    is_optimal = result['store_name'] == optimal['optimal_store']['store_name']
                    comparison_data.append({
                        'Store': result['store_name'],
                        'Time (min)': f"{result['duration']:.1f}",
                        'Distance (m)': f"{result['distance']:.0f}",
                        'Optimal': "OPTIMAL" if is_optimal else ""
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
            else:
                st.info("Only one store selected")
        
        # Comparison chart
        if len(optimal['all_results']) > 1:
            fig_comparison = px.bar(
                x=[r['store_name'] for r in optimal['all_results']],
                y=[r['duration'] for r in optimal['all_results']],
                title="Travel time comparison",
                labels={'x': 'Stores', 'y': 'Time (minutes)'},
                color=[r['duration'] for r in optimal['all_results']],
                color_continuous_scale='RdYlGn_r'
            )
            
            # Mark optimal store
            optimal_idx = next(i for i, r in enumerate(optimal['all_results']) 
                             if r['store_name'] == optimal['optimal_store']['store_name'])
            fig_comparison.add_annotation(
                x=optimal_idx,
                y=optimal['total_time'],
                text="OPTIMAL",
                showarrow=True,
                arrowhead=2,
                arrowcolor="green",
                font=dict(color="green", size=12)
            )
            
            fig_comparison.update_layout(height=400)
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    # User instructions
    if not st.session_state.selected_coords and not selected_stores:
        st.info("**Instructions:**\n"
                "1. Define a starting point\n"
                "2. Select one or more Monoprix stores\n"
                "3. Click 'Calculate optimal route'")
    elif not st.session_state.selected_coords:
        st.info("Define a starting point to calculate routes")
    elif not selected_stores:
        st.info("Select at least one Monoprix store in the sidebar")
    elif not st.session_state.optimal_route:
        st.info("Click 'Calculate optimal route' to view the simulation")

if __name__ == "__main__":
    main()
