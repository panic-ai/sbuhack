from typing import List
import uuid
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from pydantic import BaseModel
from pymongo import MongoClient
from passlib.context import CryptContext
from gridfs import GridFS
from bson import ObjectId
from fastapi.responses import StreamingResponse
import sys
sys.path.append('./models/')
from User import User
from Login import Login
from Item import Item
from oai import text_desc

from cloth_detection import complete_process    

# MongoDB connection details
MONGO_USERNAME = "sriharshapy"
MONGO_PASSWORD = "tOw5zBtJ3u3N7j7H"
MONGO_HOST = "9wpdgnl.mongodb.net"
MONGO_PORT = 27017
DATABASE_NAME = "userdb"
COLLECTION_NAME = "users"

# MongoDB connection URI with authentication
MONGO_URI = "mongodb+srv://sriharshapy:tOw5zBtJ3u3N7j7H@panicai.9wpdgnl.mongodb.net/?retryWrites=true&w=majority"

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


app = FastAPI()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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
        filelist, categorylist, colorlist = complete_process(image.file)
        text_desc = text_desc(image.file)
        for i in range(len(filelist)):
        
            create_item(username, categorylist[i], text_desc, colorlist[i], filelist[i])
    return "Yolo"


def create_item(unser_name, item_type, item_description, item_colour, files):
    if not collection.find_one({"username": unser_name}):
        raise HTTPException(status_code=404, detail="User not found")
    
    # Store file in MongoDB GridFS
    file_id=None
    for file in files:
        contents = file.read()
        file_id = fs.put(contents, filename=file.filename)

    # Insert item details into the database
    result = collectionItems.insert_one({
        "username": unser_name,
        "item_type": item_type,
        "item_description": item_description,
        "item_colour": item_colour,
        "file_id": str(file_id)  # Store the file ID in the database
    })

    if result.inserted_id:
        return {"message": "Item created successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to create item")


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

    if not collection.find_one({"username": username}):
        raise HTTPException(status_code=404, detail="User not found")    

    result = collectionItems.find({
        "username": username
    })
    resp = []
    for res in result:
        resp.append(Item(
            username= username,
            item_type= res['item_type'],
            item_description= res['item_description'],
            item_colour= res['item_colour'],
            file_id= res['file_id']
        ))
        
    return resp

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