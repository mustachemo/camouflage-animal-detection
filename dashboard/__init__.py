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
)
import pandas as pd
from .layout import layout
import cv2
import plotly.express as px
import numpy as np
import base64

app = Dash(__name__, suppress_callback_exceptions=True)
app.layout = layout


def decode_image(image_data):
    encoded_data = image_data.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.callback(Output("output-image", "figure"), [Input("upload-data", "contents")])
def update_output(contents):
    if contents is not None:
        image = decode_image(contents)
        fig = px.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return fig
    return {}


if __name__ == "__main__":
    app.run(debug=True)
