from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from detection import teeth_detection
import torch
from preprocess import preprocess_image
from utils import get_model, process_prediction, byte_to_pixels, draw_bboxes, base64encoding
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io


# File is a class that inherits directly from Form 

app = FastAPI()

model = get_model()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"details": "Tooth Diagnosis and Treatment Framework"}


@app.post('/inference')
async def infer_image(
    image_files: list[UploadFile] = File(...) # key should be image_files in form-data
):
    output = {}
    
    for image in image_files:        

        image_bytes = await image.read() # read the image as bytes .read() is an async function
        np_image = byte_to_pixels(image_bytes=image_bytes)

        # get teeth coords
        teeth_coords = teeth_detection([np_image])[0] # get all coords for teeth in the image

        # apply preprocessing to get coords
        img_data, boxes = preprocess_image(image=np_image, boxes=teeth_coords.obb.xywhr)

        # if there are no detected bounding box
        if len(boxes) == 0:
            output[image.filename] = base64encoding(np_image)
            continue

        x, y, w, h, r = zip(*boxes)

        # infer all data
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(img_data))
            conditions, treatments = process_prediction(preds)

            # annotate image
            output_image = draw_bboxes(np_image, x, y, w, h, r, conditions=conditions, treatments=treatments)
        
        output[image.filename] = base64encoding(output_image)

    return output
