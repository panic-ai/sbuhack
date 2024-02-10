from pydantic import BaseModel
class User(BaseModel):
    name: str
    email: str
    phone_number: str
    country: str
    date_of_birth: str
    username: str
    password: str
