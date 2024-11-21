from dash import html, dcc
import dash_mantine_components as dmc
from dash_iconify import DashIconify


# Define the AppShell Header
header = dmc.AppShellHeader(
    children=[
        dmc.Center(
            dmc.Text(
                "Team 69 - Camouflage Animal Detection",
                size="xl",
                style={"color": "#FFFFFF"},
            ),
            style={
                "width": "100%",
                "height": "100%",
            },
        )
    ],
    withBorder=True,
    visibleFrom={
        "xs": False,
        "sm": True,
    },  # Hides header on extra small screens, visible from small screens upwards
    zIndex=1000,  # Ensure header is on top of other content
    style={"backgroundColor": "#1A1B1E"},  # Dark background to match dark theme
)


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
        dmc.AppShell(
            [
                header,
                dmc.AppShellMain(
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "row",
                                "width": "100%",
                            },
                            children=[
                                dmc.Flex(
                                    direction="column",
                                    align="center",
                                    justify="center",
                                    children=[
                                        dmc.Center(
                                            children=[
                                                dcc.Upload(
                                                    id="upload-image",
                                                    children=dmc.Button(
                                                        "Upload Image",
                                                        style={
                                                            "backgroundColor": "#0C7FDA",
                                                            "width": "100%",
                                                        },
                                                    ),
                                                )
                                            ],
                                            style={"padding": "1rem"},
                                        ),
                                        dmc.Flex(
                                            direction="row",
                                            align="center",
                                            justify="center",
                                            children=[
                                                dmc.Loader(
                                                    children=[
                                                        html.Div(
                                                            id="output-original-image",
                                                            style={
                                                                "border": "1px solid #0C7FDA",
                                                                "width": "512px",
                                                                "height": "512px",
                                                            },
                                                        ),
                                                    ],
                                                ),
                                                dmc.Loader(
                                                    children=[
                                                        html.Div(
                                                            id="output-mask-image",
                                                            style={
                                                                "border": "1px solid #0C7FDA",
                                                                "width": "512px",
                                                                "height": "512px",
                                                            },
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="output-predicted-label",
                                            style={
                                                "textAlign": "center",
                                                "color": "#0C7FDA",
                                                "fontSize": "20px",
                                                "fontWeight": "bold",
                                                "margin": "0 20px",
                                                "alignSelf": "center",
                                            },
                                        ),
                                        html.Div(
                                            id="animal-info",
                                            style={
                                                "margin": "2rem",
                                                "padding": "10px",
                                                "border": "1px solid #ccc",
                                                "borderRadius": "5px",
                                                "backgroundColor": "#f9f9f9",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "100%",
                                    },
                                ),
                            ],
                        ),
                    ]
                ),
            ],
            header={"height": 70},
            # padding="xl",
        )
    ],
    forceColorScheme="dark",
)
