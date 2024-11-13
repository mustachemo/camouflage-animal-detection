from dash import html, dcc
import dash_mantine_components as dmc
from dash_iconify import DashIconify


# Define the AppShell Header
header = dmc.AppShellHeader(
    children=[
        dmc.Container(
            fluid=True,
            children=[
                dmc.Group(
                    align="center",
                    children=[
                        dmc.Text(
                            children="App Title",
                            id="app-title",
                            size="lg",  # Font size set to large
                            inherit=False,  # Does not inherit font properties from parent
                            inline=True,  # Sets line-height to 1 for centered alignment
                            truncate={
                                "side": "end"
                            },  # Truncates long text from the end
                        ),
                        dmc.Group(
                            children=[
                                dmc.Anchor(
                                    href="/home",
                                    children=[
                                        DashIconify(icon="tabler:home", width=24),
                                        dmc.Text("Home", size="sm"),
                                    ],
                                ),
                                dmc.Anchor(
                                    href="/about",
                                    children=[
                                        DashIconify(
                                            icon="tabler:info-circle", width=24
                                        ),
                                        dmc.Text("About", size="sm"),
                                    ],
                                ),
                                dmc.Anchor(
                                    href="/gallery",
                                    children=[
                                        DashIconify(icon="tabler:photo", width=24),
                                        dmc.Text("Gallery", size="sm"),
                                    ],
                                ),
                                dmc.Button(
                                    "Profile",
                                    variant="outline",
                                    color="indigo",
                                    id="profile-button",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
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
                                # "margin": "1rem",
                                "border": "1px solid #0f0f0f",
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
                                                    "width": "100%",
                                                },
                                            ),
                                        ),
                                        html.Div(
                                            id="output-image-upload",
                                        ),
                                    ],
                                    style={
                                        "width": "100%",
                                        "border": "1px solid #0C7FDA",
                                    },
                                ),
                                # Add other content here if needed
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
