# Dental Diagnosis and Treatment Framework
This code is the deployment code for the deep learning model developed in this [repo](https://github.com/akvnn/guiding-neural-nets). The deployment was done using Python's FastAPI, Docker, and Hugging Face Spaces.

# 1. API Description
To use the API, you may use the web interface at this [Link](https://www.diagnosemyteeth.com/). 

Or you could make the request directly using this endpoint:
```
https://Kareem-404-Tooth-Diagnosis-and-Treatment-Framework.hf.space/inference
```
 Simply, you send a POST request is sent where the images are in body's form-data where key is image_files and and value is the images. Please note that your request might take a while to process since we are using the free tier in hugging face to host the application.
# 2. How to Run Locally 
Clone the repo:
```
git clone https://github.com/Kareem404/dental-diagnosis-treatment-framework.git
```
After cloning the repo, you could run the API as a FastAPI application using uvicorn or as a docker container. 
## 2.1 Run as FastAPI App
To run the application as a FastAPI application. First, install the requirements file:
```
pip install -r requirements.txt
```
Second, run using uvicorn:
```
uvicorn aoo:app --host 0.0.0.0 --port 8000
````
This runs the application at `http://localhost:8000`. 
## 2.2 🐋 Run as Docker Container
To run the app as a docker container. First, build a docker image:
```
docker build -t dental-diagnosis-app .
```
Once the image is built, you could run it like this:
```
docker run -it -p 7860:7860 dental-diagnosis-app
```
The app should be accessible at `http://localhost:7680`
