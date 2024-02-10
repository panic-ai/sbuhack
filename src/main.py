import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from passlib.context import CryptContext
import sys
sys.path.append('./models/')
from User import User
from Login import Login

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
    

