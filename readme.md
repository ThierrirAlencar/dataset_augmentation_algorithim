
The following algorithim runs any YOLO model and generates a .csv file with the classes and detections to produce a dataset for textual context for images in a Computer Vision cenário

this code uses [moondream](moondreamv2) to generate a textual description of the image.

# Intended Uses
This dataset generation tool has the objective of generate a textual correlation between a classical Computer vision tecnique in YOLO and the textual description of the image, it can be used for NLP or image-to-text projects that require a pre made relationship between the detection data and the textual context of the image. 

This tool was firstly made to do the dataset collection of an Acne description tool based on provided an specific data, the same context can be applied by the user and/or configured for enhancements. 

# Building

Configure .env
``` Configure .env

IMAGE_DIR="images/"

MODEL_DIR="modeloM.pt"

CSV_DIR="results.csv"
```

Setup venv

````
python -m venv .venv
````

Install dependencies

````
pip install -r requirements.txt
````
