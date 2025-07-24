import streamlit as st
import pandas as pd
import pydeck as pdk

def show_species_map(species_data, species_name=None):
    """Display interactive species distribution map without IUCN status"""
    st.subheader("Species Distribution Map")
    
    if not species_data:
        st.warning("No distribution data available")
        return
    
    points = species_data.get("points", [])
    
    if not points:
        st.warning("No location data for this species")
        return
    
    df = pd.DataFrame(points)
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["lon", "lat"],
        get_color=[200, 30, 0, 160],
        get_radius="count * 10000",
        pickable=True
    )
    
    view_state = pdk.ViewState(
        latitude=df["lat"].mean(),
        longitude=df["lon"].mean(),
        zoom=2,
        pitch=0
    )
    
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "Count: {count}"}
    ))