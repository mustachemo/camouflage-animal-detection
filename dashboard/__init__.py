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

_dash_renderer._set_react_version("18.2.0")

app = Dash(
    __name__, suppress_callback_exceptions=True, external_stylesheets=dmc.styles.ALL
)
app.layout = layout

# Initialize segmentation model
birefnet, device, transform_image = initialize_seg_model()

# Helper function to sanitize filename
def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

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
    Input("upload-image", "contents"),
    State("upload-image", "filename")
)
def output_clipped_image_and_prediction(content, filename):
    if content is not None:
        # Ensure the temp directory exists
        if not os.path.exists("temp"):
            os.makedirs("temp")
        
        # Sanitize the filename
        filename = sanitize_filename(filename)

        # Decode the uploaded image and save it temporarily
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        image_path = os.path.join("temp", filename)
        with open(image_path, "wb") as f:
            f.write(decoded)

        # Generate the mask using the segmentation model
        image = Image.open(BytesIO(decoded))
        mask = get_mask(image, birefnet, device, transform_image)
        
        # Save the mask temporarily
        mask_path = os.path.join("temp", f"mask_{filename}")
        mask.save(mask_path)

        # Run classification with the original image and the mask, which also saves the clipped image
        predicted_label = predict_clipped_object(image_path, mask_path)

        # Load the clipped image (saved as "clipped_object.png" in predict_clipped_object)
        clipped_image_path = "clipped_object.png"
        with open(clipped_image_path, "rb") as f:
            clipped_image_data = f.read()
        
        # Encode the clipped image for display
        clipped_image_base64 = base64.b64encode(clipped_image_data).decode('utf-8')
        clipped_image_data_url = f"data:image/png;base64,{clipped_image_base64}"

        # Display the clipped image and the prediction result
        clipped_image_display = dmc.AspectRatio(
            dmc.Image(src=clipped_image_data_url),
            ratio=1024 / 1024,
            mx="auto",
        )
        prediction_display = dmc.Text(
            f"Prediction: {predicted_label}",
            style={"color": "#0C7FDA", "fontSize": "24px", "fontWeight": "bold"},
        )

        return clipped_image_display, prediction_display  # Return clipped image and prediction display

    return None, None

if __name__ == "__main__":
    # Ensure the temp directory exists
    if not os.path.exists("temp"):
        os.makedirs("temp")
    app.run(debug=True)
