"""
API 요청 모델들
"""
from pydantic import BaseModel
from typing import List, Optional

class ImageUrlRequest(BaseModel):
    image_url: str

class DogCreateRequest(BaseModel):
    name: str
    breed: str
    age: int
    gender: str
    size: str
    weight: Optional[str] = None
    location: str
    description: Optional[str] = None
    image_url: Optional[str] = None
    additional_images: List[str] = []
    health_info: Optional[str] = None
    vaccination: Optional[str] = None
    neutered: bool = False
    contact: Optional[str] = None
    contact_name: Optional[str] = None
    shelter_name: Optional[str] = None
    adoption_status: str = "입양 가능"
