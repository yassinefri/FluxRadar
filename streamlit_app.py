import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import networkx as nx
import requests
import time
import streamlit.components.v1 as components
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
        
        # Check that columns exist
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
    """Creates an interactive map with animation - style similar to the photo"""
    # Center on Paris
    center_lat, center_lon = 48.8566, 2.3522
    
    fig = go.Figure()
    
    # ONLY selected stores (not all stores) - Navy blue color like in the photo
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
            
            # Add ONLY selected Monoprix stores - Navy blue like in photo
            if not filtered_stores.empty:
                fig.add_trace(go.Scattermap(
                    lat=filtered_stores['latitude'],
                    lon=filtered_stores['longitude'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='#1f3a93',  # Navy blue like in the photo
                        symbol='circle'
                    ),
                    text=filtered_stores['name'] + '<br>' + filtered_stores['address'],
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    name='Selected Monoprix'
                ))
    
    # Add starting point - Green like in photo
    if selected_coords:
        fig.add_trace(go.Scattermap(
            lat=[selected_coords[0]],
            lon=[selected_coords[1]],
            mode='markers',
            marker=dict(
                size=15,
                color='#2e8b57',  # Green like in photo
                symbol='circle'
            ),
            text='Starting point',
            hovertemplate='<b>Starting point</b><extra></extra>',
            name='Start'
        ))
    
    # Add optimal route if available - Blue line like in photo
    if optimal_route and 'route_coords' in optimal_route:
        route_coords = optimal_route['route_coords']
        if route_coords and len(route_coords) > 1:
            lats, lons = zip(*route_coords)
            
            # Route line - Blue like in photo
            fig.add_trace(go.Scattermap(
                lat=lats,
                lon=lons,
                mode='lines',
                line=dict(
                    width=3,
                    color='#4285f4'  # Blue like in photo
                ),
                name='Optimal Route',
                hovertemplate='Optimal route<extra></extra>'
            ))
            
            # Create animation frames for moving point
            frames = []
            num_points = len(route_coords)
            
            # Create more frames for smoother animation (every 3rd point or at least 50 frames)
            step = max(1, num_points // 50)  # At least 50 frames for smooth animation
            
            # Create frames with moving point along the route
            for i in range(0, num_points, step):
                # Current position of animated point
                current_lat, current_lon = route_coords[i]
                
                # Create frame data - copy all existing traces and add moving point
                frame_traces = []
                
                # Add stores
                if selected_stores and not filtered_stores.empty:
                    frame_traces.append(go.Scattermap(
                        lat=filtered_stores['latitude'],
                        lon=filtered_stores['longitude'],
                        mode='markers',
                        marker=dict(size=12, color='#1f3a93', symbol='circle'),
                        text=filtered_stores['name'] + '<br>' + filtered_stores['address'],
                        hovertemplate='<b>%{text}</b><extra></extra>',
                        name='Selected Monoprix'
                    ))
                
                # Add starting point
                if selected_coords:
                    frame_traces.append(go.Scattermap(
                        lat=[selected_coords[0]],
                        lon=[selected_coords[1]],
                        mode='markers',
                        marker=dict(size=15, color='#2e8b57', symbol='circle'),
                        text='Starting point',
                        hovertemplate='<b>Starting point</b><extra></extra>',
                        name='Start'
                    ))
                
                # Add route line
                frame_traces.append(go.Scattermap(
                    lat=lats,
                    lon=lons,
                    mode='lines',
                    line=dict(width=3, color='#4285f4'),
                    name='Optimal Route',
                    hovertemplate='Optimal route<extra></extra>'
                ))
                
                # Add animated moving point - YELLOW (jaune)
                frame_traces.append(go.Scattermap(
                    lat=[current_lat],
                    lon=[current_lon],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='#FFD700',  # YELLOW/GOLD color
                        symbol='circle'
                    ),
                    text='Delivery Vehicle',
                    hovertemplate='<b>Delivery Vehicle</b><extra></extra>',
                    name='Vehicle'
                ))
                
                frames.append(go.Frame(
                    data=frame_traces,
                    name=f"frame{i}"
                ))
            
            # Add animation frames to figure
            fig.frames = frames
            
            # Add initial animated point at start - YELLOW
            fig.add_trace(go.Scattermap(
                lat=[lats[0]],
                lon=[lons[0]],
                mode='markers',
                marker=dict(
                    size=12,
                    color='#FFD700',  # YELLOW moving point
                    symbol='circle'
                ),
                text='Delivery Vehicle',
                hovertemplate='<b>Delivery Vehicle</b><extra></extra>',
                name='Vehicle'
            ))
            
            # Configure automatic looping animation (NO BUTTONS)
            fig.update_layout(
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'visible': False,  # Hide the buttons
                    'buttons': [{
                        'label': 'Auto',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 150, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 50},
                            'mode': 'immediate'
                        }]
                    }]
                }],
                # Trigger automatic animation on load
                annotations=[{
                    'text': '',
                    'showarrow': False,
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0,
                    'y': 0
                }]
            )
    
    # Map style similar to photo (light/minimal) with Paris zoom limits
    fig.update_layout(
        map=dict(
            style="carto-positron",  # Light style similar to photo
            center=dict(lat=center_lat, lon=center_lon),
            zoom=11,  # Default zoom level for Paris
            # Bounding box to limit map view to Paris region
            bounds=dict(
                west=2.224199,   # Left boundary (longitude)
                east=2.469920,   # Right boundary (longitude) 
                south=48.815573, # Bottom boundary (latitude)
                north=48.902156  # Top boundary (latitude)
            )
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.02,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        # Add zoom control limits
        uirevision='constant'  # Prevents map reset on rerun
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
    """Calculate optimal route through ALL selected stores (TSP solution)"""
    if G is None or not selected_stores:
        return None
    
    try:
        # Get addresses of selected stores
        selected_addresses = []
        for store in selected_stores:
            if " - " in store:
                address = store.split(" - ", 1)[1].strip()
                selected_addresses.append(address)
        
        filtered_stores = stores_df[stores_df['address'].isin(selected_addresses)]
        
        if filtered_stores.empty:
            return None
        
        # If only one store, return simple route
        if len(filtered_stores) == 1:
            store = filtered_stores.iloc[0]
            dest_coords = (store['latitude'], store['longitude'])
            metrics = calculate_route_metrics(G, start_coords, dest_coords, avg_speed_kmh)
            if metrics:
                return {
                    'stores_order': [store['address']],
                    'route_coords': metrics['route_coords'],
                    'total_time': metrics['duration'],
                    'total_distance': metrics['distance'],
                    'segments': [{
                        'from': 'Starting Point',
                        'to': store['address'],
                        'distance': metrics['distance'],
                        'duration': metrics['duration']
                    }]
                }
            return None
        
        # Multiple stores - solve TSP
        stores_list = filtered_stores.to_dict('records')
        
        # Calculate distance matrix between all points (start + all stores)
        all_points = [start_coords] + [(store['latitude'], store['longitude']) for store in stores_list]
        n_points = len(all_points)
        
        # Distance matrix calculation
        distance_matrix = [[0] * n_points for _ in range(n_points)]
        time_matrix = [[0] * n_points for _ in range(n_points)]
        
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    metrics = calculate_route_metrics(G, all_points[i], all_points[j], avg_speed_kmh)
                    if metrics:
                        distance_matrix[i][j] = metrics['distance']
                        time_matrix[i][j] = metrics['duration']
                    else:
                        # If route calculation fails, use high penalty
                        distance_matrix[i][j] = float('inf')
                        time_matrix[i][j] = float('inf')
        
        # Solve TSP using nearest neighbor heuristic (good for small number of stores)
        def solve_tsp_nearest_neighbor(matrix, start_idx=0):
            n = len(matrix)
            visited = [False] * n
            tour = [start_idx]
            visited[start_idx] = True
            current = start_idx
            total_cost = 0
            
            # Visit all other nodes
            for _ in range(n - 1):
                nearest_dist = float('inf')
                nearest_idx = -1
                
                for j in range(n):
                    if not visited[j] and matrix[current][j] < nearest_dist:
                        nearest_dist = matrix[current][j]
                        nearest_idx = j
                
                if nearest_idx != -1:
                    tour.append(nearest_idx)
                    visited[nearest_idx] = True
                    total_cost += matrix[current][nearest_idx]
                    current = nearest_idx
            
            # RETURN TO STARTING POINT (close the loop)
            tour.append(start_idx)
            total_cost += matrix[current][start_idx]
            
            return tour, total_cost
        
        # Find optimal tour (minimize travel time)
        optimal_tour, total_time = solve_tsp_nearest_neighbor(time_matrix, 0)
        
        # Calculate total distance for the optimal tour
        total_distance = sum(distance_matrix[optimal_tour[i]][optimal_tour[i+1]] 
                           for i in range(len(optimal_tour)-1))
        
        # Build the complete route coordinates (including return to start)
        complete_route_coords = []
        segments = []
        
        for i in range(len(optimal_tour) - 1):
            from_idx = optimal_tour[i]
            to_idx = optimal_tour[i + 1]
            
            from_coords = all_points[from_idx]
            to_coords = all_points[to_idx]
            
            segment_metrics = calculate_route_metrics(G, from_coords, to_coords, avg_speed_kmh)
            if segment_metrics and segment_metrics['route_coords']:
                # Add segment coordinates (skip first point to avoid duplication)
                if i == 0:
                    complete_route_coords.extend(segment_metrics['route_coords'])
                else:
                    complete_route_coords.extend(segment_metrics['route_coords'][1:])
                
                # Store segment info
                if from_idx == 0:
                    from_name = "Starting Point"
                elif to_idx == 0:  # Returning to start
                    from_name = stores_list[from_idx-1]['address']
                else:
                    from_name = stores_list[from_idx-1]['address']
                
                if to_idx == 0:  # Returning to start
                    to_name = "Starting Point"
                else:
                    to_name = stores_list[to_idx-1]['address']
                
                segments.append({
                    'from': from_name,
                    'to': to_name,
                    'distance': segment_metrics['distance'],
                    'duration': segment_metrics['duration']
                })
        
        # Store order (excluding starting point, but noting it returns)
        stores_order = [stores_list[idx-1]['address'] for idx in optimal_tour[1:-1]]  # Exclude start and final return
        
        return {
            'stores_order': stores_order,
            'route_coords': complete_route_coords,
            'total_time': total_time,
            'total_distance': total_distance,
            'segments': segments,
            'tour_indices': optimal_tour
        }
    
    except Exception as e:
        st.error(f"Error calculating optimal route: {str(e)}")
        return None

# Address geocoding functions
def geocode_address(address):
    """Convert address to coordinates using Nominatim (OpenStreetMap)"""
    try:
        # Use Nominatim API (free, no API key required)
        base_url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': address,
            'format': 'json',
            'limit': 1,
            'countrycodes': 'fr',  # Focus on France
            'addressdetails': 1
        }
        
        # Add user agent to be respectful to the API
        headers = {
            'User-Agent': 'FluxRadar-DeliveryApp/1.0'
        }
        
        response = requests.get(base_url, params=params, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                result = data[0]
                lat = float(result['lat'])
                lon = float(result['lon'])
                display_name = result.get('display_name', address)
                return lat, lon, display_name
        
        return None, None, None
    except Exception as e:
        st.error(f"Error geocoding address: {str(e)}")
        return None, None, None

def search_addresses_realtime(query):
    """Real-time address search for Paris with Nominatim"""
    if not query or len(query) < 2:
        return []
    
    try:
        # Multiple search strategies for better results
        search_queries = [
            f"{query}, Paris, France",  # Most specific
            f"{query} Paris",           # Less specific
            f"{query}"                  # Let API decide
        ]
        
        all_suggestions = []
        
        for search_query in search_queries:
            base_url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': search_query,
                'format': 'json',
                'limit': 5,
                'countrycodes': 'fr',
                'addressdetails': 1,
                'bounded': 1,  # Only within bounding box
                'viewbox': '2.224199,48.902156,2.469920,48.815573',  # Paris bounding box
            }
            
            headers = {
                'User-Agent': 'FluxRadar-DeliveryApp/1.0'
            }
            
            response = requests.get(base_url, params=params, headers=headers, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                for item in data:
                    display_name = item.get('display_name', '')
                    # Only keep Paris addresses
                    if 'Paris' in display_name:
                        # Clean up the display name
                        parts = display_name.split(', ')
                        if len(parts) >= 2:
                            # Try to format nicely
                            street = parts[0]
                            if len(parts) >= 3 and 'Arrondissement' in parts[1]:
                                clean_name = f"{street}, {parts[1]}, Paris"
                            elif len(parts) >= 2:
                                clean_name = f"{street}, Paris"
                            else:
                                clean_name = display_name.replace(', France', '')
                            
                            all_suggestions.append(clean_name)
            
            # If we found good results, break early
            if len(all_suggestions) >= 6:
                break
            
            # Small delay between requests to be respectful
            time.sleep(0.1)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for item in all_suggestions:
            if item not in seen:
                seen.add(item)
                unique_suggestions.append(item)
        
        return unique_suggestions[:8]  # Limit to 8 suggestions
        
    except Exception:
        return []

def search_addresses_french_api(query):
    """Use French government API for better address search"""
    if not query or len(query) < 3:
        return []
    
    try:
        # Use the official French address API (much better for France)
        base_url = "https://api-adresse.data.gouv.fr/search/"
        params = {
            'q': query,
            'limit': 8,
            'type': 'housenumber',
            'postcode': '750*',  # Paris postcodes (75001-75020)
        }
        
        response = requests.get(base_url, params=params, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            suggestions = []
            
            for feature in data.get('features', []):
                properties = feature.get('properties', {})
                label = properties.get('label', '')
                
                # Only keep Paris addresses
                if '75' in label and 'Paris' in label:
                    suggestions.append(label)
            
            return suggestions[:8]
        
        return []
    except Exception:
        return []

def search_addresses_multiple_apis(query):
    """Try multiple APIs for best results"""
    if not query or len(query) < 2:
        return []
    
    all_suggestions = []
    
    # Try French API first
    french_results = search_addresses_french_api(query)
    all_suggestions.extend(french_results)
    
    # If we don't have enough results, try Nominatim
    if len(all_suggestions) < 5:
        nominatim_results = search_addresses_realtime(query)
        all_suggestions.extend(nominatim_results)
    
    # Remove duplicates
    seen = set()
    unique_suggestions = []
    for item in all_suggestions:
        if item not in seen:
            seen.add(item)
            unique_suggestions.append(item)
    
    return unique_suggestions[:8]
    """Real-time address search for Paris with autocomplete"""
    if not query or len(query) < 2:
        return []
    
    try:
        # Multiple search strategies for better results
        search_queries = [
            f"{query}, Paris, France",  # Most specific
            f"{query} Paris",           # Less specific
            f"{query}"                  # Let API decide
        ]
        
        all_suggestions = []
        
        for search_query in search_queries:
            base_url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': search_query,
                'format': 'json',
                'limit': 5,
                'countrycodes': 'fr',
                'addressdetails': 1,
                'bounded': 1,  # Only within bounding box
                'viewbox': '2.224199,48.902156,2.469920,48.815573',  # Paris bounding box
            }
            
            headers = {
                'User-Agent': 'FluxRadar-DeliveryApp/1.0'
            }
            
            response = requests.get(base_url, params=params, headers=headers, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                for item in data:
                    display_name = item.get('display_name', '')
                    # Only keep Paris addresses
                    if 'Paris' in display_name:
                        # Clean up the display name
                        parts = display_name.split(', ')
                        if len(parts) >= 2:
                            # Try to format nicely
                            street = parts[0]
                            if len(parts) >= 3 and 'Arrondissement' in parts[1]:
                                clean_name = f"{street}, {parts[1]}, Paris"
                            elif len(parts) >= 2:
                                clean_name = f"{street}, Paris"
                            else:
                                clean_name = display_name.replace(', France', '')
                            
                            all_suggestions.append(clean_name)
            
            # If we found good results, break early
            if len(all_suggestions) >= 6:
                break
            
            # Small delay between requests to be respectful
            time.sleep(0.1)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for item in all_suggestions:
            if item not in seen:
                seen.add(item)
                unique_suggestions.append(item)
        
        return unique_suggestions[:8]  # Limit to 8 suggestions
        
    except Exception:
        return []

def search_addresses(query):
    """Search for addresses with autocomplete"""
    if len(query) < 3:  # Only search if query is at least 3 characters
        return []
    
    try:
        base_url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': query,
            'format': 'json',
            'limit': 5,
            'countrycodes': 'fr',
            'addressdetails': 1
        }
        
        headers = {
            'User-Agent': 'FluxRadar-DeliveryApp/1.0'
        }
        
        response = requests.get(base_url, params=params, headers=headers, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            suggestions = []
            for item in data:
                display_name = item.get('display_name', '')
                # Clean up the display name for better readability
                if ', France' in display_name:
                    display_name = display_name.replace(', France', '')
                suggestions.append(display_name)
            return suggestions
        
        return []
    except Exception:
        return []

def render_address_input():
    """Render simple address input with manual search"""
    st.markdown("### 📍 Enter Starting Address")
    
    col_input, col_button = st.columns([3, 1])
    
    with col_input:
        address_manual = st.text_input(
            "Manual address search",
            placeholder="Ex: 15 Rue de Rivoli, Paris",
            key="manual_address_input",
            help="Enter an address and click 'Set Starting Point'"
        )
    
    with col_button:
        st.write("") # Spacing
        if st.button("🎯 Set Starting Point", type="primary", help="Set this address as starting point"):
            if address_manual:
                with st.spinner("📍 Geocoding in progress..."):
                    lat, lon, display_name = geocode_address(address_manual)
                    if lat and lon:
                        st.session_state.selected_coords = (lat, lon)
                        st.session_state.selected_address = display_name
                        st.success(f"✅ Starting point set: **{display_name}**")
                        st.rerun()
                    else:
                        st.error("❌ Unable to find this address. Please check spelling.")
            else:
                st.warning("⚠️ Please enter an address first.")

def search_addresses_enhanced(query):
    """Enhanced address search combining multiple APIs with better error handling"""
    if not query or len(query) < 2:
        return []
    
    try:
        all_suggestions = []
        
        # French government API - more accurate for French addresses
        try:
            french_url = f"https://api-adresse.data.gouv.fr/search/"
            params = {
                'q': f"{query}, Paris",
                'limit': 5,
                'type': 'housenumber',
                'autocomplete': 1
            }
            
            response = requests.get(french_url, params=params, timeout=3)
            if response.status_code == 200:
                data = response.json()
                for feature in data.get('features', []):
                    props = feature.get('properties', {})
                    # Check if it's in Paris (postal code starts with 75)
                    if props.get('context') and '75' in props.get('context', ''):
                        address = f"{props.get('name', '')} {props.get('street', '')}".strip()
                        if address:
                            full_address = f"{address}, {props.get('city', 'Paris')}"
                            all_suggestions.append(full_address)
        except Exception as e:
            st.warning(f"French API error: {e}")
        
        # Nominatim as backup
        if len(all_suggestions) < 3:
            try:
                nominatim_url = "https://nominatim.openstreetmap.org/search"
                params = {
                    'q': f"{query}, Paris, France",
                    'format': 'json',
                    'limit': 5,
                    'addressdetails': 1,
                    'bounded': 1,
                    'viewbox': '2.224199,48.902156,2.469920,48.815573',
                }
                
                headers = {'User-Agent': 'FluxRadar-DeliveryApp/1.0'}
                response = requests.get(nominatim_url, params=params, headers=headers, timeout=3)
                
                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        display_name = item.get('display_name', '')
                        if 'Paris' in display_name:
                            parts = display_name.split(', ')
                            clean_name = f"{parts[0]}, Paris" if len(parts) >= 2 else display_name
                            all_suggestions.append(clean_name)
            except Exception as e:
                st.warning(f"Nominatim API error: {e}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for item in all_suggestions:
            if item not in seen:
                seen.add(item)
                unique_suggestions.append(item)
        
        return unique_suggestions[:6]
        
    except Exception as e:
        st.error(f"Error searching addresses: {e}")
        return []

def add_auto_animation_script():
    """Add JavaScript to automatically start and loop the animation"""
    script = """
    <script>
    // Wait for Plotly to load, then start animation automatically
    setTimeout(function() {
        // Find all Plotly graphs on the page
        var plots = document.querySelectorAll('.js-plotly-plot');
        plots.forEach(function(plot) {
            if (plot && plot.data && plot.layout && plot.layout.updatemenus) {
                // Start the animation automatically and set it to loop
                Plotly.animate(plot, null, {
                    frame: {duration: 150, redraw: true},
                    transition: {duration: 50},
                    mode: 'immediate'
                }).then(function() {
                    // After animation ends, restart it (infinite loop)
                    setInterval(function() {
                        if (plot && plot.data) {
                            Plotly.animate(plot, null, {
                                frame: {duration: 150, redraw: true},
                                transition: {duration: 50},
                                mode: 'immediate'
                            });
                        }
                    }, plot.frames ? plot.frames.length * 150 + 1000 : 10000); // Restart after all frames + 1sec
                });
            }
        });
    }, 1000); // Wait 1 second for Plotly to fully load
    </script>
    """
    components.html(script, height=0)

# ...existing code...

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
        
        # State to store selected coordinates and address
        if 'selected_coords' not in st.session_state:
            st.session_state.selected_coords = None
        if 'selected_address' not in st.session_state:
            st.session_state.selected_address = None
        if 'last_selected_address' not in st.session_state:
            st.session_state.last_selected_address = ''
        
        # Enhanced address selection interface
        st.markdown("## 🚀 Select Starting Point")
        
        # Simple address input interface
        render_address_input()
        
        # Show current selected address
        if st.session_state.get('selected_address') and st.session_state.selected_coords:
            st.success(f"Current starting point: **{st.session_state.selected_address}**")
            if st.button("Clear starting point"):
                # Clear all address-related session state
                for key in ['selected_coords', 'selected_address', 'last_selected_address', 'address_input']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
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
                        num_stores = len(optimal_route['stores_order'])
                        if num_stores == 1:
                            st.success(f"Route calculated to: {optimal_route['stores_order'][0]}")
                        else:
                            st.success(f"Optimal route calculated through {num_stores} stores")
                            with st.expander("Store visiting order"):
                                for i, store in enumerate(optimal_route['stores_order'], 1):
                                    st.write(f"{i}. {store}")
                        st.info(f"Total time: {optimal_route['total_time']:.1f} min | Total distance: {optimal_route['total_distance']:.0f}m | Speed: {avg_speed} km/h")
                    else:
                        st.error("Unable to calculate optimal route")
        
        # Display map with optimal route
        fig = create_paris_map(stores_df, st.session_state.selected_coords, 
                              selected_stores, st.session_state.optimal_route)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add auto-animation script if there's a route with animation
        if st.session_state.optimal_route and 'route_coords' in st.session_state.optimal_route:
            if st.session_state.optimal_route['route_coords']:
                add_auto_animation_script()
    
    with col2:
        st.subheader("Information")
        
        # Store statistics
        st.metric("Number of Monoprix", len(stores_df))
        st.metric("Selected stores", len(selected_stores))
        
        # Optimal route information
        if st.session_state.optimal_route:
            optimal = st.session_state.optimal_route
            st.success("Optimal route calculated")
            
            num_stores = len(optimal['stores_order'])
            if num_stores == 1:
                st.metric("Destination", optimal['stores_order'][0])
            else:
                st.metric("Stores to visit", num_stores)
            
            st.metric("Total time", f"{optimal['total_time']:.1f} min")
            st.metric("Total distance", f"{optimal['total_distance']:.0f} m")
            
            if st.button("Clear route"):
                st.session_state.optimal_route = None
                st.rerun()
        else:
            if st.session_state.selected_coords:
                st.success("Starting point defined")
            else:
                st.info("Define a starting point")
    
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