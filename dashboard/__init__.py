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
import dash_mantine_components as dmc

_dash_renderer._set_react_version("18.2.0")

app = Dash(
    __name__, suppress_callback_exceptions=True, external_stylesheets=dmc.styles.ALL
)
app.layout = layout


@callback(
    Output("output-image-upload", "children"),
    Input("upload-image", "contents"),
)
def output_uploaded_image(content):
    return dmc.AspectRatio(
        dmc.Image(src=content),
        ratio=1,
        mx="auto",
    )


if __name__ == "__main__":
    app.run(debug=True)
