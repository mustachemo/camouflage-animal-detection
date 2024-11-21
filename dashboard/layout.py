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
    },
    zIndex=1000,
    style={
        "backgroundColor": "#1e90ff",
        "padding": "10px",
        "boxShadow": "0 0 3px rgba(0, 0, 0, 0.5)",
        "borderBottomLeftRadius": "20px",
        "borderBottomRightRadius": "20px",
    },
)

# Define the layout
layout = dmc.MantineProvider(
    id="mantine-provider",
    theme={
        "primaryColor": "indigo",
        "fontFamily": "'Inter', sans-serif",
    },
    children=[
        dmc.AppShell(
            children=[
                header,
                dmc.Flex(
                    direction="row",
                    justify="center",
                    align="center",
                    style={"padding": "20px", "gap": "20px"},
                    children=[
                        # Drag and Drop Upload Section
                        dmc.Paper(
                            style={
                                "width": "45%",
                                "padding": "20px",
                                "border": "1px dashed #1E90FF",
                                "borderRadius": "10px",
                                "textAlign": "center",
                                "backgroundColor": "#F8FAFC",
                            },
                            children=[
                                DashIconify(icon="bi:cloud-upload", width=50, color="#1E90FF"),
                                dmc.Text("Drag and Drop files to upload", fw=500, size="lg"),
                                dmc.Text("or", size="sm"),
                                dcc.Upload(
                                    id="upload-image",
                                    children=dmc.Button(
                                        "Browse",
                                        style={
                                            "backgroundColor": "#1E90FF",
                                            "color": "#FFFFFF",
                                            "borderRadius": "5px",
                                            "padding": "10px 20px",
                                        },
                                    ),
                                    style={"marginTop": "10px"},
                                ),
                                dmc.Text(
                                    "Supported files: PNG, JPG",
                                    size="xs",
                                    style={"color": "#6c757d"},
                                ),
                            ],
                        ),
                        # Uploaded and Segmentation Outputs
                        dmc.Paper(
                            style={
                                "width": "90%",
                                "padding": "20px",
                                "borderRadius": "10px",
                                "backgroundColor": "#FFFFFF",
                                "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.5)",
                                "marginTop": "50px",
                            },
                            children=[
                                html.Div(
                                    style={"display": "flex", "gap": "10px", "justifyContent": "center"},
                                    children=[
                                        html.Div(
                                            id="output-original-image",
                                            style={
                                                "padding": "10px",
                                                "border": "1px solid #ccc",
                                                "width": "400px",
                                                "height": "400px",
                                                "display": "flex",
                                                "alignItems": "center",
                                                "justifyContent": "center",
                                                "backgroundColor": "#f9f9f9",
                                                "boxShadow": "0 0 6px rgba(0, 0, 0, 0.5)",
                                            },
                                        ),
                                        # Add Loading Spinner to Mask Image Container
                                        dcc.Loading(
                                            id="loading-mask",
                                            type="circle",  # Spinner type: circle
                                            color="#1E90FF",  # Spinner color
                                            children=html.Div(
                                                id="output-mask-image",
                                                style={
                                                    "border": "1px solid #ccc",
                                                    "width": "400px",
                                                    "height": "400px",
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "justifyContent": "center",
                                                    "backgroundColor": "#f9f9f9",
                                                    "boxShadow": "0 0 6px rgba(0, 0, 0, 0.5)",
                                                },
                                            ),
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
                                        "margin": "20px 0",
                                    },
                                ),
                                html.Div(
                                    id="animal-info",
                                    style={
                                        "marginTop": "20px",
                                        "padding": "10px",
                                        "border": "1px solid #ccc",
                                        "borderRadius": "5px",
                                        "backgroundColor": "#f9f9f9",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)
