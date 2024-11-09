from dash import html, dcc, dash_table
import dash_mantine_components as dmc
from dash_iconify import DashIconify

layout = dmc.MantineProvider(
    forceColorScheme="dark",
    children=[
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
        ),
        html.Div([
            dcc.Loading(
                id="output-image",
                type="circle",
                children=[html.Div([dcc.Graph(id="output-image")])],
            )
        ]),
    ],
)
