import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from utils.detection import detect_camouflaged_animals
from utils.video_processing import extract_frames
import cv2
from dotenv import load_dotenv
import os
import requests

load_dotenv()
app = dash.Dash(__name__)
mapbox_access_token = os.getenv("MAPBOX_ACCESS_TOKEN")

app.layout = html.Div([
    html.H1("Camouflaged Object Detection", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
    dcc.Upload(
        id='upload-image',
        children=html.Div(['Drag and Drop or ', html.A('Select Image', style={'color': '#3498db'})]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '2px', 'borderStyle': 'dashed',
            'borderRadius': '10px', 'textAlign': 'center', 'margin': '20px 0',
            'backgroundColor': '#ecf0f1', 'color': '#7f8c8d'
        },
        multiple=False
    ),
    dcc.Graph(id='output-image', style={'border': '1px solid #bdc3c7', 'borderRadius': '10px', 'padding': '10px'}),
    html.Div(id='map-label', style={'textAlign': 'center', 'marginTop': '20px', 'fontSize': '20px', 'color': '#2c3e50'}),
    dcc.Graph(id='map', style={'border': '1px solid #bdc3c7', 'borderRadius': '10px', 'padding': '10px', 'marginTop': '10px'}),
], style={'maxWidth': '800px', 'margin': 'auto', 'padding': '20px', 'fontFamily': 'Arial, sans-serif'})

def fetch_gbif_data(query):
    url = f"https://api.gbif.org/v1/occurrence/search?q={query}&limit=1000"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        occurrences = data.get('results', [])
        latitudes = [occ['decimalLatitude'] for occ in occurrences if 'decimalLatitude' in occ]
        longitudes = [occ['decimalLongitude'] for occ in occurrences if 'decimalLongitude' in occ]
        return latitudes, longitudes
    else:
        return [], []

def create_map_figure(latitudes, longitudes):
    fig = go.Figure(go.Scattermapbox(
        lat=latitudes,
        lon=longitudes,
        mode='markers',
        marker=go.scattermapbox.Marker(size=9),
        text=["Occurrence"] * len(latitudes)
    ))
    
    fig.update_layout(
        mapbox={
            'accesstoken': mapbox_access_token,
            'style': "open-street-map",
            'center': {"lat": 0, "lon": 0},
            'zoom': 1
        },
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    
    return fig

# Set default data as "frog"
default_latitudes, default_longitudes = fetch_gbif_data("frog")

# Callback for image upload and detection
@app.callback(
    Output('output-image', 'figure'),
    [Input('upload-image', 'contents')]
)
def update_output(contents):
    if contents is not None:
        image, result = detect_camouflaged_animals(contents)
        fig = px.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        return fig
    return {}

# Callback to update the map and label
@app.callback(
    [Output('map', 'figure'), Output('map-label', 'children')],
    [Input('upload-image', 'contents')]
)
def update_map(contents):
    if contents is not None:
        classification = "frog"  # Replace with actual classification logic (split the classification label at the "-" and take the second part)
        
        latitudes, longitudes = fetch_gbif_data(classification)
        label = f"Showing the last {len(latitudes)} occurrences of {classification}"
        return create_map_figure(latitudes, longitudes), label
    
    # Default data for "frog"
    label = f"Showing the last {len(default_latitudes)} occurrences of frog"
    return create_map_figure(default_latitudes, default_longitudes), label

if __name__ == '__main__':
    app.run_server(debug=True)