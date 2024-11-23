# Camouflaged Object Detection System

This project is an AI-powered system designed to detect camouflaged animals in images and videos. It integrates object detection using a fine-tuned **Segment Anything Model (SAM2)**, species classification, and video frame analysis. The application is aimed at wildlife researchers, conservationists, and technology enthusiasts interested in advancements in AI and computer vision.

## Features

- **Image Detection**: Detect camouflaged animals in still images using advanced segmentation models.
- **Species Classification**: Classify the detected animals into various species.
- **Informational Output**: Provide detailed information about the detected animal's habitat and species characteristics.
- **Interactive Visualization**: Display the detection results with segmentation masks, classification labels, and any other relevant information.

## Technology Stack

- **Backend Framework**: Dash (based on Flask)
- **Machine Learning Model**: BiRefNet for segmentation and CNN-based models for classification.
- **Data Processing**: NumPy, OpenCV for video and image processing.
- **Visualization**: Plotly, integrated with Dash for displaying results, including image overlays and detection boundaries.
- **Map API**: MapBox https://docs.mapbox.com/help/getting-started/access-tokens/ (note you will need to create an account to generate a token, it is free)

## Directory Structure

```bash
## Directory Structure

```bash
├── run.py                      # Main entry point for the Dash web app
├── configs/                    # Configuration files
│   └── Species_Labels_1.csv    # CSV file with species labels
├── dashboard/                  # Dash application files
│   ├── __init__.py             # Initialization file for the Dash app
│   └── layout.py               # Layout definition for the Dash app
├── data/                       # Data files and scripts
│   ├── bootstrap_camo.sh       # Script to bootstrap camo data
│   ├── bootstrap_cod10k.sh     # Script to bootstrap COD10K data
│   ├── README.md               # README for data directory
│   ├── resize_images.py        # Script to resize images
│   ├── sample_files/           # Sample files for testing
│   ├── test/                   # Test dataset
│   └── train/                  # Training dataset
├── Dockerfile                  # Docker configuration file
├── fine_tuned_model.pth        # Fine-tuned model file
├── logs/                       # Logs directory
│   ├── classification_resnet_metrics_Validation.csv  # Validation metrics for classification
│   ├── classification_resnet_metrics.json            # JSON file with classification metrics
│   └── segmentation_model_finetune.csv               # CSV file with segmentation model finetune metrics
├── models/                     # Model files
│   ├── seg_model.py            # Segmentation model script
│   └── test_seg.py             # Script to test segmentation model
├── README.md                   # Project overview and setup instructions
├── requirements.txt            # Dependencies and libraries
├── sam2/                       # SAM2 model files
│   └── demo/                   # Demo files for SAM2 model
├── temp/                       # Temporary files
│   ├── config.json             # Configuration file
│   └── predicted_label.txt     # Predicted label file
└── utils/                      # Utility functions
    ├── classification.py       # Classification logic
    ├── detection.py            # Image detection logic
    ├── video_processing.py     # Scripts for extracting frames from videos
    └── transform_data.py       # Data transformation script
```

## Setup Instructions

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/camouflaged-object-detection.git
   cd camouflaged-object-detection
   ```

2. Create a file named ".env" in the root directory of the repo. In the .env file, add the following line:

   ``` bash
   MAPBOX_ACCESS_TOKEN=<YOUR_MAPBOX_TOKEN_HERE> (with no quotes)
    ```
   
   It should look like this: 

   ``` bash
   MAPBOX_ACCESS_TOKEN=pk.124135435...
   ```

4. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the Dash web server:

   ```bash
   python app.py
   ```

6. Open the app by navigating to `http://localhost:8080/` in your browser.

### Docker Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/camouflaged-object-detection.git
   cd camouflaged-object-detection
   ```

2. Create a file named ".env" in the root directory of the repo. In the .env file, add the following line:

   ``` bash
   MAPBOX_ACCESS_TOKEN=<YOUR_MAPBOX_TOKEN_HERE> (with no quotes)
    ```
   
   It should look like this: 

   ``` bash
   MAPBOX_ACCESS_TOKEN=pk.124135435...
   ```

3. Build the Docker image:

   ```bash
   docker build -t camouflaged-object-detection .
   ```

4. Run the Docker container:

   ```bash
    docker run -p 8080:8080 -v ${PWD}/:/app --gpus all camouflaged-object-detection
    ```

5. Open the app by navigating to `http://localhost:8080/` in your browser.


## Bonus Features

- **Improve Real-Time Performance**: Optimize the model for real-time detection in live video feeds.
- **Expand Species Database**: Incorporate more species to improve classification accuracy.
- **Interactive Visualizations**: Enable more detailed interactive graphs and information displays for the users.
