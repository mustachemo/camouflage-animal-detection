from dash import (
    Dash,
    Input,
    Output,
    State,
    callback,
    callback_context,
    exceptions,
    dcc,
    html,
    exceptions,
    DiskcacheManager,
    no_update,
    _dash_renderer,
)
import base64
import os
from PIL import Image
from io import BytesIO
import pandas as pd
import plotly.graph_objects as go  # Import this for Scattermapbox and figure creation
from .layout import layout,fetch_gbif_data, create_map_figure
import cv2
import plotly.express as px
import numpy as np
import dash_mantine_components as dmc
from models.seg_model import initialize_seg_model, get_mask
from utils.classification import predict_clipped_object  # Import predict_clipped_object
import re
import requests

_dash_renderer._set_react_version("18.2.0")

app = Dash(
    __name__, suppress_callback_exceptions=True, external_stylesheets=dmc.styles.ALL
)
app.layout = layout

# Initialize segmentation model
birefnet, device, transform_image = initialize_seg_model()


def get_animal_info(animal_name):
    # Attempt to fetch information from Wikidata
    wikidata_info = get_animal_info_from_wikidata(animal_name)
    if wikidata_info:
        return format_info_wikidata(animal_name, wikidata_info)

    # Fallback to Wikipedia API if Wikidata information is not available
    url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&titles={animal_name.replace(' ', '_')}&prop=extracts|pageprops&explaintext&redirects=1"
    response = requests.get(url)
    data = response.json()

    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})

    if "extract" in page:
        return format_info_wikipedia(page["title"], page["extract"])
    else:
        return f"No information found for {animal_name}."


def get_animal_info_from_wikidata(animal_name):
    query = f"""
    SELECT ?item ?itemLabel ?nativeHabitatLabel ?size WHERE {{
      ?item rdfs:label "{animal_name}"@en.
      OPTIONAL {{ ?item wdt:P183 ?nativeHabitat. }}
      OPTIONAL {{ ?item wdt:P2048 ?size. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/json"}
    response = requests.get(url, params={"query": query}, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if "results" in data.get("results", {}) and data["results"]["bindings"]:
            return data["results"]["bindings"]
    return None


def format_info_wikidata(animal_name, wikidata_data):
    # Extract information from Wikidata response
    native_habitat = next(
        (
            item.get("nativeHabitatLabel", {}).get("value", "Not available.")
            for item in wikidata_data
        ),
        "Not available.",
    )
    size = next(
        (item.get("size", {}).get("value", "Not available.") for item in wikidata_data),
        "Not available.",
    )

    # Prepare HTML output
    return html.Div([
        html.H4(animal_name),
        html.P(f"Summary: Information retrieved from Wikidata."),
        html.P(f"Native Habitat: {native_habitat}"),
        html.P(f"Size: {size}"),
    ])


def format_info_wikipedia(title, extract):
    # Extract key sections from Wikipedia extract
    origin = re.search(
        r"(origin:.*?\.|native to.*?\.|found in.*?\.|inhabits.*?\.|distributed in.*?\.|occurs in.*?\.|endemic to.*?\.|lives in.*?\.)",
        extract,
        re.IGNORECASE,
    )
    behavior = re.search(
        r"(behavior:.*?\.|known for.*?\.|feeds on.*?\.|hunts.*?\.|is nocturnal.*?\.|is diurnal.*?\.|exhibits.*? behavior.*?\.|mating.*?\.|social behavior.*?\.)",
        extract,
        re.IGNORECASE,
    )
    size = re.search(
        r"(size:.*?\.|can grow up to.*?\.|weighs up to.*?\.|length.*?\.|height.*?\.|mass.*?\.|can reach.*?\.|measures.*?\.)",
        extract,
        re.IGNORECASE,
    )

    # Get the first part of the extract as a summary
    summary = extract.split(".")[0].strip() + "."

    # Prepare HTML output
    return html.Div(
        [
            html.P(f"Summary: {summary}"),  # Summary at the beginning
            html.P(
                f"Origin: {origin.group(0).strip() if origin else 'Not available.'}"
            ),
            html.P(
                f"Behavior: {behavior.group(0).strip() if behavior else 'Not available.'}"
            ),
            html.P(f"Size: {size.group(0).strip() if size else 'Not available.'}"),
        ],
        style={
            "padding": "1rem",
            "borderRadius": "5px",
            "backgroundColor": "#f9f9f9",
        },
    )


# Helper function to sanitize filename
def sanitize_filename(filename):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)


@callback(
    Output("output-original-image", "children"),
    Input("upload-image", "contents"),
)
def output_original_image(content):
    if content is not None:
        return dmc.AspectRatio(
            dmc.Image(src=content),
            ratio=1024 / 1024,
            mx="auto",
        )


@callback(
    Output("output-mask-image", "children"),
    Output("output-predicted-label", "children"),  # Add output for predicted label
    Output("animal-info", "children"),  # New output for animal information
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
)
def output_clipped_image_and_prediction(content, filename):
    if content is not None:
        if not os.path.exists("temp"):
            os.makedirs("temp")

        filename = sanitize_filename(filename)
        content_type, content_string = content.split(",")
        decoded = base64.b64decode(content_string)
        image_path = os.path.join("temp", filename)
        with open(image_path, "wb") as f:
            f.write(decoded)

        image = Image.open(BytesIO(decoded))
        mask = get_mask(image, birefnet, device, transform_image)

        mask_path = os.path.join("temp", f"mask_{filename}")
        mask.save(mask_path)

        predicted_label = predict_clipped_object(image_path, mask_path)

        clipped_image_path = "clipped_object.png"
        with open(clipped_image_path, "rb") as f:
            clipped_image_data = f.read()

        clipped_image_base64 = base64.b64encode(clipped_image_data).decode("utf-8")
        clipped_image_data_url = f"data:image/png;base64,{clipped_image_base64}"

        clipped_image_display = dmc.AspectRatio(
            dmc.Image(src=clipped_image_data_url),
            ratio=1024 / 1024,
            mx="auto",
        )

        class_num, animal_type = predicted_label

        prediction_display = dmc.Text(
            f"Prediction: {animal_type}",
            style={"color": "#0C7FDA", "fontSize": "24px", "fontWeight": "bold"},
        )

        animal_name = str(animal_type.split("-")[1]).lower()

        # Fetch animal information
        animal_info = get_animal_info(animal_name)

        return (
            clipped_image_display,
            prediction_display,
            animal_info,
        )  # Include animal info

    return None, None, None

# Fetch data from GBIF
def fetch_gbif_data(query):
    url = f"https://api.gbif.org/v1/occurrence/search?q={query}&limit=1000"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        occurrences = data.get("results", [])
        latitudes = [occ['decimalLatitude'] for occ in occurrences if 'decimalLatitude' in occ]
        longitudes = [occ['decimalLongitude'] for occ in occurrences if 'decimalLongitude' in occ]
        return latitudes, longitudes
    else:
        return [], []

# Create map figure
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
            'accesstoken': os.getenv("MAPBOX_ACCESS_TOKEN"),
            'style': "open-street-map",
            'center': {"lat": 0, "lon": 0},
            'zoom': 1,
        },
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )
    return fig

@callback(
    [Output("map", "figure"), 
     Output("map-label", "children")],
    [Input("output-predicted-label", "children")],
)
def update_map(predicted_label):
    # Check if predicted_label is a dictionary (e.g., a Dash component like dmc.Text)
    if isinstance(predicted_label, dict) and "props" in predicted_label:
        # Extract the actual label from the `children` property
        label_text = predicted_label["props"].get("children", "")
    elif isinstance(predicted_label, str):
        # If it's already a string, use it directly
        label_text = predicted_label
    else:
        return no_update, "No prediction available to generate map."

    # Ensure the label contains "Prediction:"
    if "Prediction:" in label_text:
        try:
            # Extract the animal name from the label
            full_label = str(label_text.split(":")[1]).strip()
            animal_name = full_label.split("-")[1] if "-" in full_label else full_label

            # Fetch GBIF data for the specific animal name
            latitudes, longitudes = fetch_gbif_data(animal_name)

            if latitudes and longitudes:
                label = f"Showing the last {len(latitudes)} occurrences of {animal_name}"
            else:
                label = f"No occurrence data available for {animal_name}"

            return create_map_figure(latitudes, longitudes), label

        except Exception as e:
            print(f"Error processing predicted label: {e}")
            return no_update, "Error processing map data."
    
    print("Predicted label does not contain 'Prediction:'.")
    return no_update, "No prediction available to generate map."


if __name__ == "__main__":
    # Ensure the temp directory exists
    if not os.path.exists("temp"):
        os.makedirs("temp")
    app.run(debug=True)
