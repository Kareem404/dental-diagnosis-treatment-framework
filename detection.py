from ultralytics import YOLO
import ultralytics
import numpy as np
from huggingface_hub import hf_hub_download
from utils import hf_token

YOLO_file = hf_hub_download(repo_id="Kareem-404/dental-diagnosis-and-treatment", 
                            filename="best.pt", 
                            token=hf_token)
# get the model
model = YOLO(YOLO_file)

def teeth_detection(images: list[np.array]) -> ultralytics.engine.results.Results:
    return model.predict(images)