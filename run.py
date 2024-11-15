import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from utils.detection import detect_camouflaged_animals
from utils.video_processing import extract_frames
import cv2
import requests
import re

app = dash.Dash(__name__)

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
], style={'maxWidth': '800px', 'margin': 'auto', 'padding': '20px', 'fontFamily': 'Arial, sans-serif'}), html.Div([
    dcc.Input(id='animal-input', type='text', placeholder='Enter animal name'),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='output-container')
])

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

# Text box input
@app.callback(
    Output('output-container', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('animal-input', 'value')
)
def update_output(n_clicks, animal_name):
    if n_clicks > 0 and animal_name:
        return get_animal_info(animal_name)
    return "Enter an animal name and press submit."


def get_animal_info(animal_name):
    url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&titles={animal_name.replace(' ', '_')}&prop=extracts|pageprops&explaintext&redirects=1"
    response = requests.get(url)
    data = response.json()

    pages = data['query']['pages']
    page = next(iter(pages.values()))

    if 'extract' in page:
        return format_info(page['title'], page['extract'])
    else:
        return f"No information found for {animal_name}."


def format_info(title, extract):
    # Extract relevant information
    origin = re.search(r'origin.*?(\.|$)', extract, re.IGNORECASE)
    behavior = re.search(r'behavior.*?(\.|$)', extract, re.IGNORECASE)
    size = re.search(r'size.*?(\.|$)', extract, re.IGNORECASE)
    fun_fact = "Did you know? " + extract.split('.')[0] + "."

    return html.Div([
        html.H4(title),
        html.P(f"Origin: {origin.group(0).strip() if origin else 'Not available.'}"),
        html.P(f"Behavior: {behavior.group(0).strip() if behavior else 'Not available.'}"),
        html.P(f"Size: {size.group(0).strip() if size else 'Not available.'}"),
        html.P(f"Other Relevant Facts: {fun_fact}")
    ])


if __name__ == '__main__':
    app.run_server(debug=True)
