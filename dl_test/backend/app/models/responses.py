"""
API 응답 모델들
"""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class SearchResult(BaseModel):
    rank: int
    image_url: Optional[str]
    keypoint_image_path: Optional[str]
    simclr_similarity: float
    keypoint_similarity: float
    combined_similarity: float
    db_info: Dict[str, Any]
    keypoint_processing_time: Optional[float] = None

class SearchMetadata(BaseModel):
    database_size: int
    images_with_data: int
    searched_results: int
    confidence_threshold: float
    algorithm: str
    processing_time: float
    model_version: str
    feature_dimension: int

class SearchResponse(BaseModel):
    success: bool
    query_image: str
    query_keypoint_image: Optional[str]
    results: List[SearchResult]
    mode: str
    search_metadata: SearchMetadata
    error: Optional[str] = None
    message: Optional[str] = None

class FeatureExtractionResponse(BaseModel):
    status: str
    feature_vector: Optional[List[float]] = None
    vector_dimension: Optional[int] = None
    filename: Optional[str] = None
    image_url: Optional[str] = None
    image_size_bytes: Optional[int] = None
    content_type: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    models_available: bool
    ap10k_model_loaded: bool
    mode: str
    simclr_model_path: Optional[str] = None
    message: str
