# Camouflaged Object Detection System

This project is an AI-powered system designed to detect camouflaged animals in images and videos. It integrates object detection using a fine-tuned **Segment Anything Model (SAM2)**, species classification, and video frame analysis. The application is aimed at wildlife researchers, conservationists, and technology enthusiasts interested in advancements in AI and computer vision.

## IMPORT INFORMATION
- in order for SAM2 to work it needs the config files. Having the config files in sam2/configs/ doesn't work for some reason, I got over this hurdle by having it in the root directory of the project.


## Features

- **Image Detection**: Detect camouflaged animals in still images using advanced segmentation models.
- **Species Classification**: Classify the detected animals into various species.
- **Video Analysis**: Extract frames from YouTube videos or pass in a video and analyze them/it for camouflaged animals.
- **Informational Output**: Provide detailed information about the detected animal's habitat and species characteristics.
- **Interactive Visualization**: Display the detection results with bounding boxes, segmentation masks, classification labels, and any other relevant information.
- **Performance Metrics**: Measure accuracy with precision, recall, and F1-score.

## Technology Stack

- **Backend Framework**: Dash (based on Flask)
- **Machine Learning Model**: Segment Anything Model (SAM2) for segmentation and CNN-based models for classification.
- **Data Processing**: NumPy, OpenCV for video frame extraction and image processing.
- **Visualization**: Plotly, integrated with Dash for displaying results, including image overlays and detection boundaries.

## Directory Structure

```bash
├── app.py                    # Main entry point for the Dash web app
├── models/                   # Pretrained SAM2 models and fine-tuned classification models
├── static/                   # Static files (e.g., images, CSS)
│   └── sample_data/          # Sample images for testing
├── templates/                # HTML templates for Dash layouts
├── utils/                    # Utility functions for image processing, classification, etc.
│   ├── video_processing.py   # Scripts for extracting frames from videos
│   └── detection.py          # Image detection and classification logic
├── data/                     # Datasets for training and evaluation
├── README.md                 # Project overview and setup instructions
└── requirements.txt          # Dependencies and libraries
```

## Setup Instructions

### Prerequisites

- **Python 3.8+**
- **Dash**: Install via `pip`
- **Other Dependencies**:
  - OpenCV
  - TensorFlow or PyTorch (depending on model framework)
  - Plotly
  - NumPy
  - SAM2 Model (ensure it is available or download via the model repository)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/camouflaged-object-detection.git
   cd camouflaged-object-detection
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Download and place the pretrained models in the `models/` directory.

#### download sam2
   
follow the steps from: 
   ```bash
   https://github.com/facebookresearch/sam2
   ```

5. Start the Dash web server:

   ```bash
   python app.py
   ```

5. Open the app by navigating to `http://localhost:8050/` in your browser.

### Docker Setup

1. Build the Docker image:

   ```bash
   docker build -t camouflaged-object-detection .
   ```

2. Run the Docker container:

   ```bash
    docker run -p 8050:8050 camouflaged-object-detection
    ```

3. Open the app by navigating to `http://localhost:8050/` in your browser.


## Bonus Features

- **Improve Real-Time Performance**: Optimize the model for real-time detection in live video feeds.
- **Expand Species Database**: Incorporate more species to improve classification accuracy.
- **Interactive Visualizations**: Enable more detailed interactive graphs and information displays for the users.