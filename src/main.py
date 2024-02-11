from typing import List
import uuid
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from pydantic import BaseModel
from pymongo import MongoClient
from passlib.context import CryptContext
from gridfs import GridFS
from bson import ObjectId
import requests
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from fastapi.responses import StreamingResponse
import sys
from starlette.staticfiles import StaticFiles
from starlette.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
sys.path.append('./models/')
from User import User
from Login import Login
from Item import Item
from oai import text_desc
from fastapi.responses import FileResponse
from pathlib import Path
from cloth_detection import save_segmented_parts
from PIL import Image
import io
import os
import json
from torchvision.transforms import ToTensor, ToPILImage


# MongoDB connection details
MONGO_USERNAME = "sriharshapy"
MONGO_PASSWORD = "V9KLsQBdhZxGm9cK"
MONGO_HOST = "9wpdgnl.mongodb.net"
MONGO_PORT = 27017
DATABASE_NAME = "userdb"
COLLECTION_NAME = "users"

MODEL_NAME = "valentinafeve/yolos-fashionpedia"
feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
model = YolosForObjectDetection.from_pretrained(MODEL_NAME)

# MongoDB connection URI with authentication
MONGO_URI = "mongodb+srv://sriharshapy:V9KLsQBdhZxGm9cK@cluster0.d4ts45w.mongodb.net/?retryWrites=true&w=majority"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

dbFilesClient = client["userFiles"]
collectionFiles = dbFilesClient["files"]
fs = GridFS(dbFilesClient)

dbItemsDetails = client["items"]
collectionItems = dbItemsDetails["item"]

# Define Pydantic model for user registration

origins = ["*"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def complete_process(image):
    image = fix_channels(ToTensor()(image))
    print ("Fix channels done")
    inputs = feature_extractor(images=image, return_tensors="pt")
    print ("Feature extraction done")
    outputs = model(**inputs)
    print ("Output done")
    return save_segmented_parts(image, outputs, threshold=0.5)

def fix_channels(t):
    """
    Some images may have 4 channels (transparent images) or just 1 channel (black and white images), in order to let the images have only 3 channels. I am going to remove the fourth channel in transparent images and stack the single channel in back and white images.
    :param t: Tensor-like image
    :return: Tensor-like image with three channels
    """
    if len(t.shape) == 2:
        return ToPILImage()(torch.stack([t for i in (0, 0, 0)]))
    if t.shape[0] == 4:
        return ToPILImage()(t[:3])
    if t.shape[0] == 1:
        return ToPILImage()(torch.stack([t[0] for i in (0, 0, 0)]))
    return ToPILImage()(t)

@app.get("/clozy/UI/{imageasset}")
async def get_image_assets(imageasset: str):
    # Check if the provided file ID is valid
        file_path = "UI/images/"+imageasset
        image_path = Path(file_path)
        if not image_path.is_file():
            raise HTTPException(status_code=400, detail="no such image : "+imageasset)
        return FileResponse(image_path)


@app.get("/clozy", response_class=HTMLResponse)
async def serving_index():
    # Path to your HTML file inside the 'static' directory
    file_path = "UI/index.html"

    # Read the HTML file
    with open(file_path, "r") as file:
        html_content = file.read()

    return HTMLResponse(content=html_content, status_code=200)


@app.get("/clozy/signup", response_class=HTMLResponse)
async def serving_signup():
    # Path to your HTML file inside the 'static' directory
    file_path = "UI/signup.html"

    # Read the HTML file
    with open(file_path, "r") as file:
        html_content = file.read()

    return HTMLResponse(content=html_content, status_code=200)

@app.get("/clozy/homepage", response_class=HTMLResponse)
async def serving_signup():
    # Path to your HTML file inside the 'static' directory
    file_path = "UI/homepage.html"

    # Read the HTML file
    with open(file_path, "r") as file:
        html_content = file.read()

    return HTMLResponse(content=html_content, status_code=200)


@app.post("/register/")
async def register_user(user: User):
    # Check if the username or email already exists
    existing_user = collection.find_one({"$or": [{"username": user.username}, {"email": user.email}]})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    if user.gender not in ["male","female","non binary"]:
        raise HTTPException(status_code=400, detail="Invalid gender")

    # Insert the new user into the database with the unique ObjectId
    result = collection.insert_one({
        "name": user.name,
        "email": user.email,
        "phone_number": user.phone_number,
        "country": user.country,
        "date_of_birth": user.date_of_birth,
        "username": user.username,
        "password": pwd_context.hash(user.password) ,
        "gender" : user.gender
    })
    if result.inserted_id:
        # Get the ObjectId of the inserted document
        user_id = str(result.inserted_id)
        return {"message": "User registered successfully", "user_id": user_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to register user")

@app.post("/login/")
async def login_user(login: Login):
    user = collection.find_one({"username": login.username})
    if user:
        if pwd_context.verify(login.password, user["password"]):
            return {"message": "Login successful"}
        else:
            raise HTTPException(status_code=401, detail="Incorrect password")
    else:
        raise HTTPException(status_code=404, detail="User not found")

@app.post("/processImages/")
async def processImages(username: str = Form(...),
                        images:List[UploadFile] = File(...)):
    for image in images:
        contents = await image.read()
        # Create a bytes buffer from the file content
        bytes_io = io.BytesIO(contents)
        print ("\nContent rceived\n")

        # Open the bytes buffer with PIL
        pil_image = Image.open(bytes_io)
        fiximage = fix_channels(ToTensor()(pil_image))
        print ("Fix channels done")
        inputs = feature_extractor(images=fiximage, return_tensors="pt")
        print ("Feature extraction done")
        outputs = model(**inputs)
        print ("Output done")
        filelist, categorylist, colorlist = save_segmented_parts(fiximage, outputs, threshold=0.5)
        print ("complete_process complete")
        print ("\n\nfilelist \n",filelist)
        print ("\n\ncategorylist\n ",categorylist)
        print ("\n\ncolorlist\n ",colorlist)
        text_desctiption = text_desc(bytes_io)
        print ("\n\text_desctiption\n ",text_desctiption)
        for i in range(len(filelist)):
            location = "/home/ubuntu/images/"+username+"/"
            if not os.path.exists(location):
                os.makedirs(location)
            category = categorylist[i].split(',')
            temp = ''
            for i in category:
                temp=temp+"_"+i
            fname = location+"-"+colorlist[i]+"-"+temp
            filelist[i].save(fname+"-"+image.filename)
            data = {
                "location": location,
                "color": colorlist[i],
                "categorylist": temp
            }
            with open(location+".json", "w") as json_file:
                json.dump(data, json_file, indent=4)

# Close the image file
            filelist[i].close()
            #create_item(username, categorylist[i], text_desctiption, colorlist[i], upload_file)
    return "Yolo"



@app.post("/items/")
async def create_item(username: str = Form(...),
    item_type: str = Form(...),
    item_description: str = Form(...),
    item_colour: str = Form(...),
    files: List[UploadFile] = File(...)):

    if not collection.find_one({"username": username}):
        raise HTTPException(status_code=404, detail="User not found")

    # Store file in MongoDB GridFS
    file_id=None
    for file in files:
        contents = await file.read()
        file_id = fs.put(contents, filename=file.filename)

    # Insert item details into the database
    result = collectionItems.insert_one({
        "username": username,
        "item_type": item_type,
        "item_description": item_description,
        "item_colour": item_colour,
        "file_id": str(file_id)  # Store the file ID in the database
    })

    if result.inserted_id:
        return {"message": "Item created successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to create item")

@app.get("/getItems/{username}")
async def getItems(username: str):
    location = "/home/ubuntu/images/"+username+"/"
    files = os.listdir(location)
    line = ""
    for file in files:
        for x in  file.split(".")[0].split("-")[:-1]:
            line = line +" " +x
        line = line + ","
    return line

@app.get("/files/{file_id}")
async def get_file(file_id: str):
    # Check if the provided file ID is valid
    if not ObjectId.is_valid(file_id):
        raise HTTPException(status_code=400, detail="Invalid file ID")

    # Retrieve the file from GridFS by its ID
    file_object = fs.get(ObjectId(file_id))
    if file_object:
        # Retrieve the file name
        file_name = file_object.filename

        # Return the file contents and file name as response
        return StreamingResponse(iter([file_object.read()]), media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename={file_name}"})
    else:
        raise HTTPException(status_code=404, detail="File not found")
