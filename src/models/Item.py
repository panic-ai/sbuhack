from pydantic import BaseModel
class Item(BaseModel):
    username: str
    item_type: str
    item_description: str
    item_colour: str
    file_id: str