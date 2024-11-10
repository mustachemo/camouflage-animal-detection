from dash import html, dcc
import dash_mantine_components as dmc
from dash_iconify import DashIconify


layout = dmc.MantineProvider(
    id="mantine-provider",
    children=[
        dcc.Upload(
            id="upload-image",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "50%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "10px",
                "textAlign": "center",
                "margin": "20px",
                "backgroundColor": "#f0f0f0",
                "color": "#333",
                "fontFamily": "Arial, sans-serif",
                "fontSize": "16px",
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
            },
        ),
        html.Div(
            id="output-image-upload", style={"textAlign": "left", "margin": "20px"}
        ),
    ],
    forceColorScheme="dark",
)
