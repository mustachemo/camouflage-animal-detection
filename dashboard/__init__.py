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
import pandas as pd
from .layout import layout
import cv2
import plotly.express as px
import numpy as np
import base64
import dash_mantine_components as dmc

app = Dash(__name__, suppress_callback_exceptions=True)
app.layout = layout


@callback(
    Output("output-image-upload", "children"),
    Input("upload-image", "contents"),
)
def output_uploaded_image(content):
    return html.Img(src=content, style={"width": "50%", "height": "50%"})


if __name__ == "__main__":
    app.run(debug=True)
