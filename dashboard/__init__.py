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
from .layout import layout
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
    url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&titles={animal_name.replace(' ', '_')}&prop=extracts|pageprops&explaintext&redirects=1"
    response = requests.get(url)
    data = response.json()

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))

    if "extract" in page:
        return format_info(page["title"], page["extract"])
    else:
        return f"No information found for {animal_name}."


def format_info(title, extract):
    # Extract relevant information
    origin = re.search(r"origin.*?(\.|$)", extract, re.IGNORECASE)
    behavior = re.search(r"behavior.*?(\.|$)", extract, re.IGNORECASE)
    size = re.search(r"size.*?(\.|$)", extract, re.IGNORECASE)
    fun_fact = "Did you know? " + extract.split(".")[0] + "."

    return html.Div([
        html.H4(title),
        html.P(f"Origin: {origin.group(0).strip() if origin else 'Not available.'}"),
        html.P(
            f"Behavior: {behavior.group(0).strip() if behavior else 'Not available.'}"
        ),
        html.P(f"Size: {size.group(0).strip() if size else 'Not available.'}"),
        html.P(f"Other Relevant Facts: {fun_fact}"),
    ])


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


if __name__ == "__main__":
    # Ensure the temp directory exists
    if not os.path.exists("temp"):
        os.makedirs("temp")
    app.run(debug=True)
