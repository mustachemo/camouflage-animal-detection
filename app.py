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
    dcc.Graph(id='map', style={'border': '1px solid #bdc3c7', 'borderRadius': '10px', 'padding': '10px', 'marginTop': '20px'}),
], style={'maxWidth': '800px', 'margin': 'auto', 'padding': '20px', 'fontFamily': 'Arial, sans-serif'})

def create_map_figure(center_coords, border_coords):
    fig = go.Figure(go.Scattermapbox(
        lat=[center_coords["lat"]],
        lon=[center_coords["lon"]],
        mode='markers',
        marker=go.scattermapbox.Marker(size=14),
        text=["Location"]
    ))
    
    fig.update_layout(
        mapbox={
            'accesstoken': mapbox_access_token,
            'style': "open-street-map",
            'center': center_coords,
            'zoom': 10,
            'layers': [{
                'source': {
                    'type': "FeatureCollection",
                    'features': [{
                        'type': "Feature",
                        'geometry': {
                            'type': "Polygon",
                            'coordinates': [border_coords]
                        }
                    }]
                },
                'type': "line",
                'color': "blue"
            }]
        },
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    
    return fig

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

# Callback to update the map
@app.callback(
    Output('map', 'figure'),
    [Input('upload-image', 'contents')]
)
def update_map(contents):
    center_coords = {"lat": 33.7490, "lon": -84.3880}
    border_coords = [
        [-84.5517, 33.6475],
        [-84.5517, 33.8500],
        [-84.2249, 33.8500],
        [-84.2249, 33.6475],
        [-84.5517, 33.6475]
    ]
    
    return create_map_figure(center_coords, border_coords)

if __name__ == '__main__':
    app.run_server(debug=True)