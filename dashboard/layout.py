from dash import html, dcc
import dash_mantine_components as dmc
from dash_iconify import DashIconify


layout = dmc.MantineProvider(
    id="mantine-provider",
    theme={
        "primaryColor": "indigo",
        "fontFamily": "'Inter', sans-serif",
        "components": {
            "Button": {"defaultProps": {"fw": 400}},
            "Alert": {"styles": {"title": {"fontWeight": 500}}},
            "AvatarGroup": {"styles": {"truncated": {"fontWeight": 500}}},
            "Badge": {"styles": {"root": {"fontWeight": 500}}},
            "Progress": {"styles": {"label": {"fontWeight": 500}}},
            "RingProgress": {"styles": {"label": {"fontWeight": 500}}},
            "CodeHighlightTabs": {"styles": {"file": {"padding": 12}}},
            "Table": {
                "defaultProps": {
                    "highlightOnHover": True,
                    "withTableBorder": True,
                    "verticalSpacing": "sm",
                    "horizontalSpacing": "md",
                }
            },
        },
    },
    children=[
        dmc.Flex(
            direction="column",
            align="left",
            justify="left",
            children=[
                dcc.Upload(
                    id="upload-image",
                    children=dmc.Button(
                        "Upload Image",
                        style={
                            "backgroundColor": "#0C7FDA",
                            "marginBottom": "20px",
                        },
                    ),
                ),
                html.Div(
                    id="output-image-upload",
                    style={
                        "width": "25%",
                        "height": "25%",
                    },
                ),
            ],
            style={"margin": "20px"},
        ),
    ],
    forceColorScheme="dark",
)
