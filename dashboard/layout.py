from dash import html, dcc
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import plotly.graph_objects as go
import requests
import os
from dotenv import load_dotenv

load_dotenv()

mapbox_access_token = os.getenv("MAPBOX_ACCESS_TOKEN")

# Fetch data from GBIF
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

# Create the map figure
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
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    
    return fig

# Define the AppShell Header
header = dmc.AppShellHeader(
    children=[
        dmc.Center(
            dmc.Text(
                "Team 69 - Camouflage Animal Detection",
                size="xl",
                style={"color": "#FFFFFF"},
            ),
            style={
                "width": "100%",
                "height": "100%",
            },
        )
    ],
    withBorder=True,
    visibleFrom={
        "xs": False,
        "sm": True,
    },
    zIndex=1000,
    style={
        "backgroundColor": "#1e90ff",
        "padding": "10px",
        "boxShadow": "0 0 3px rgba(0, 0, 0, 0.5)",
        "borderBottomLeftRadius": "20px",
        "borderBottomRightRadius": "20px",
    },
)

# Define the layout
layout = dmc.MantineProvider(
    id="mantine-provider",
    theme={
        "primaryColor": "indigo",
        "fontFamily": "'Inter', sans-serif",
    },
    children=[
        dmc.AppShell(
            children=[
                header,
                # Main Content Section
                dmc.Flex(
                    direction="row",
                    justify="center",
                    align="flex-start",
                    style={"padding": "20px", "gap": "20px"},
                    children=[
                        # File Upload Section
                        dmc.Paper(
                            style={
                                "width": "45%",
                                "padding": "20px",
                                "border": "1px dashed #1E90FF",
                                "borderRadius": "10px",
                                "textAlign": "center",
                                "backgroundColor": "#F8FAFC",
                                "marginTop": "150px",
                            },
                            children=[
                                DashIconify(icon="bi:cloud-upload", width=50, color="#1E90FF"),
                                dmc.Text("Drag and Drop files to upload", fw=500, size="lg"),
                                dmc.Text("or", size="sm"),
                                dcc.Upload(
                                    id="upload-image",
                                    children=dmc.Button(
                                        "Browse",
                                        style={
                                            "backgroundColor": "#1E90FF",
                                            "color": "#FFFFFF",
                                            "borderRadius": "5px",
                                            "padding": "10px 20px",
                                        },
                                    ),
                                    style={"marginTop": "10px"},
                                ),
                                dmc.Text(
                                    "Supported files: PNG, JPG",
                                    size="xs",
                                    style={"color": "#6c757d"},
                                ),
                            ],
                        ),
                        # Segmentation Outputs Section
                        dmc.Paper(
                            style={
                                "width": "90%",
                                "padding": "20px",
                                "borderRadius": "10px",
                                "backgroundColor": "#FFFFFF",
                                "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.5)",
                                "marginTop": "50px",
                            },
                            children=[
                                html.Div(
                                    style={"display": "flex", "gap": "10px", "justifyContent": "center"},
                                    children=[
                                        html.Div(
                                            id="output-original-image",
                                            style={
                                                "padding": "10px",
                                                "border": "1px solid #ccc",
                                                "width": "400px",
                                                "height": "400px",
                                                "display": "flex",
                                                "alignItems": "center",
                                                "justifyContent": "center",
                                                "backgroundColor": "#f9f9f9",
                                                "boxShadow": "0 0 6px rgba(0, 0, 0, 0.5)",
                                            },
                                        ),
                                        dcc.Loading(
                                            id="loading-mask",
                                            type="circle",
                                            color="#1E90FF",
                                            children=html.Div(
                                                id="output-mask-image",
                                                style={
                                                    "border": "1px solid #ccc",
                                                    "width": "400px",
                                                    "height": "400px",
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "justifyContent": "center",
                                                    "backgroundColor": "#f9f9f9",
                                                    "boxShadow": "0 0 6px rgba(0, 0, 0, 0.5)",
                                                },
                                            ),
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="output-predicted-label",
                                    style={
                                        "textAlign": "center",
                                        "color": "#0C7FDA",
                                        "fontSize": "20px",
                                        "fontWeight": "bold",
                                        "margin": "20px 0",
                                    },
                                ),
                                html.Div(
                                    id="animal-info",
                                    style={
                                        "marginTop": "20px",
                                        "padding": "10px",
                                        "border": "1px solid #ccc",
                                        "borderRadius": "5px",
                                        "backgroundColor": "#f9f9f9",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                # Map Section as Separate Container
                dmc.Paper(
                    style={
                        "width": "95%",
                        "margin": "20px auto",
                        "padding": "20px",
                        "border": "1px solid #bdc3c7",
                        "borderRadius": "10px",
                        "backgroundColor": "#FFFFFF",
                        "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.5)",
                    },
                    children=[
                        dcc.Loading(
                            id="loading-map",  # Spinner for the map
                            type="circle",
                            color="#1E90FF",
                            children=dcc.Graph(
                                id="map",
                                style={
                                    "width": "100%",
                                    "height": "500px",
                                },
                            ),
                        ),
                        html.Div(
                            id="map-label",
                            style={
                                "textAlign": "center",
                                "marginTop": "10px",
                                "fontSize": "18px",
                                "color": "#2c3e50",
                            },
                        ),
                    ],
                ),
            ],
        ),
    ],
)
