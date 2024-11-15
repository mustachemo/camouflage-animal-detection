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
from segmentation.seg_model import initialize_seg_model, get_mask

_dash_renderer._set_react_version("18.2.0")


app = Dash(
    __name__, suppress_callback_exceptions=True, external_stylesheets=dmc.styles.ALL
)
app.layout = layout

birefnet, device, transform_image = initialize_seg_model()


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
    Input("upload-image", "contents"),
)
def output_mask_image(content):
    if content is not None:
        # decode the base64 image data
        content_type, content_string = content.split(",")
        decoded = base64.b64decode(content_string)
        image = Image.open(BytesIO(decoded))

        mask = get_mask(image, birefnet, device, transform_image)

        # encode the mask to base64
        buffered = BytesIO()
        mask.save(buffered, format="JPEG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        mask_data = f"data:image/jpeg;base64,{mask_base64}"

        return dmc.AspectRatio(
            dmc.Image(src=mask_data),
            ratio=1024 / 1024,
            mx="auto",
        )


if __name__ == "__main__":
    # initialize_seg_model()
    app.run(debug=True)
