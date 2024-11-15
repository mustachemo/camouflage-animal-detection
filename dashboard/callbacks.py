import tempfile
import cv2
import os
import base64
import io
from PIL import Image
from dash import Input, Output, State, callback, html
from utils.classification import predict_clipped_object  # Import your prediction function

@callback(
    [
        Output("output-classification", "children"), 
        Output("output-image-upload", "children", allow_duplicate=True)
    ],
    [Input("upload-image", "contents")],
    [State("upload-image", "filename")],
    prevent_initial_call=True
)
def process_uploaded_image(contents, filename):
    if contents is None:
        return "Please upload an image.", None

    # Decode the uploaded image
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))

    # Convert RGBA to RGB if necessary
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)  # Save the converted image to the temporary file
        tmp_file_path = tmp_file.name

    try:
        # Pass the temporary file path to your OpenCV-based prediction function
        prediction = predict_clipped_object(tmp_file_path)
    finally:
        # Clean up the temporary file after processing
        os.remove(tmp_file_path)

    prediction_text = html.Div(
        children=[
            html.H4("Prediction Result:", style={"font-size": "24px", "color": "#0C7FDA", "margin-bottom": "10px"}),
            html.P(
                f"{prediction}",
                style={
                    "font-weight": "bold",
                    "font-size": "20px",
                    "color": "#333",
                    "padding": "10px",
                    "background-color": "#f0f4f8",
                    "border": "1px solid #0C7FDA",
                    "border-radius": "5px",
                },
            ),
        ],
        style={"text-align": "center", "margin-top": "20px"}
    )

    # Display the uploaded image in output-image-upload
    image_display = html.Img(src=contents, style={"width": "100%", "height": "100%", "border": "1px solid #ddd", "border-radius": "5px"})

    # Return the classification result and image display
    return prediction_text, image_display
