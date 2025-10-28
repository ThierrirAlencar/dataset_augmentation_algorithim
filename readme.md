
The following algorithim runs any YOLO model and generates a .csv file with the classes and detections and uses [moondream](moondreamv2) to describe the image.


# Building it

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