"""
강아지 정보 관리 API 라우터
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.services import DogService
from app.models.requests import DogCreateRequest
from app.database import DatabaseManager

router = APIRouter(prefix="/api", tags=["dogs"])
dog_service = DogService()

@router.get("/dogs/")
async def get_all_dogs_api():
    """실제 DB에서 모든 강아지 정보 조회"""
    try:
        dogs = dog_service.get_all_dogs()
        
        return JSONResponse({
            "status": "success",
            "dogs": dogs,
            "total_count": len(dogs)
        })
        
    except Exception as e:
        print(f"❌ DB 조회 실패: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/dogs/{dog_id}")
async def get_dog_detail_api(dog_id: int):
    """특정 강아지 상세 정보 조회"""
    try:
        dog = dog_service.get_dog_by_id(dog_id)
        if not dog:
            return JSONResponse({
                "status": "error",
                "message": "강아지를 찾을 수 없습니다"
            }, status_code=404)
        
        return JSONResponse({
            "status": "success", 
            "dog": dog
        })
        
    except Exception as e:
        print(f"❌ 강아지 상세 정보 조회 실패: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.post("/dogs/")
async def add_dog_api(dog_data: DogCreateRequest):
    """새 강아지 정보 추가"""
    try:
        dog_id = dog_service.create_dog(dog_data.dict())
        return JSONResponse({
            "status": "success",
            "message": "강아지 정보가 추가되었습니다",
            "dog_id": dog_id
        })
    except Exception as e:
        print(f"❌ 강아지 추가 실패: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.put("/dogs/{dog_id}")
async def update_dog_api(dog_id: int, dog_data: DogCreateRequest):
    """강아지 정보 수정"""
    try:
        success = dog_service.update_dog(dog_id, dog_data.dict())
        if success:
            return JSONResponse({
                "status": "success",
                "message": "강아지 정보가 수정되었습니다"
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": "강아지를 찾을 수 없습니다"
            }, status_code=404)
    except Exception as e:
        print(f"❌ 강아지 수정 실패: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.delete("/dogs/{dog_id}")
async def delete_dog_api(dog_id: int):
    """강아지 정보 삭제"""
    try:
        success = dog_service.delete_dog(dog_id)
        if success:
            return JSONResponse({
                "status": "success",
                "message": "강아지 정보가 삭제되었습니다"
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": "강아지를 찾을 수 없습니다"
            }, status_code=404)
    except Exception as e:
        print(f"❌ 강아지 삭제 실패: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.post("/dogs/{dog_id}/image_mapping")
async def add_image_mapping_api(dog_id: int, image_path: str, feature_vector: list = None):
    """강아지-이미지 매핑 추가"""
    try:
        mapping_id = dog_service.add_image_mapping(dog_id, image_path, feature_vector)
        return JSONResponse({
            "status": "success",
            "message": "이미지 매핑이 추가되었습니다",
            "mapping_id": mapping_id
        })
    except Exception as e:
        print(f"❌ 이미지 매핑 추가 실패: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@router.get("/breed_codes/")
async def get_breed_codes_api():
    """견종 코드 목록 조회"""
    try:
        breed_codes = dog_service.get_breed_codes()
        return JSONResponse({
            "status": "success",
            "breed_codes": breed_codes,
            "total_count": len(breed_codes)
        })
    except Exception as e:
        print(f"❌ 견종 코드 조회 실패: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

# 디버그 API들
@router.get("/debug/db-tables")
async def get_db_tables():
    """데이터베이스 테이블 목록 조회 (디버그용)"""
    try:
        db = DatabaseManager()
        tables = db.show_tables()
        return {"success": True, "tables": tables}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/debug/db-table-structure/{table_name}")
async def get_table_structure(table_name: str):
    """특정 테이블의 구조 조회 (디버그용)"""
    try:
        db = DatabaseManager()
        structure = db.describe_table(table_name)
        return {"success": True, "table": table_name, "structure": structure}
    except Exception as e:
        return {"success": False, "error": str(e)}
