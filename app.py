import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from utils.detection import detect_camouflaged_animals
from utils.video_processing import extract_frames
import cv2

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Camouflaged Object Detection"),
    dcc.Upload(
        id='upload-image',
        children=html.Div(['Drag and Drop or ', html.A('Select Image')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),
    dcc.Graph(id='output-image'),
])

#Callback for image upload and detection
@app.callback(
    Output('output-image', 'figure'),
    [Input('upload-image', 'contents')]
)
def update_output(contents):
    if contents is not None:
        image, result = detect_camouflaged_animals(contents)
        fig = px.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        return fig
    return {}

if __name__ == '__main__':
    app.run_server(debug=True)
