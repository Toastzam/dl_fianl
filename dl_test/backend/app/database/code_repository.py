"""
코드 관련 데이터베이스 리포지토리
"""
from typing import Dict, List
from mysql.connector import Error

from .connection import get_shared_db_connection

class CodeRepository:
    """공통 코드 관리 리포지토리"""
    
    def __init__(self):
        self.db_connection = get_shared_db_connection()
    
    def get_breed_codes(self) -> List[Dict[str, str]]:
        """견종 코드 목록 조회"""
        try:
            with self.db_connection.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT cd, cd_nm
                    FROM cmn_code
                    WHERE group_cd = 'DOG_BREED' AND use_yn = 'Y'
                    ORDER BY cd_nm
                """)
                breed_codes = cursor.fetchall()
                print(f"✅ DOG_BREED 코드 {len(breed_codes)}개 조회")
                return breed_codes
        except Error as e:
            print(f"❌ DOG_BREED 코드 조회 실패: {e}")
            return []
    
    def get_breed_name_by_code(self, breed_code: str) -> str:
        """견종 코드로 견종 이름 조회"""
        if not breed_code:
            return '정보 없음'
        try:
            with self.db_connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT cd_nm
                    FROM cmn_code
                    WHERE group_cd = 'DOG_BREED' AND cd = %s AND use_yn = 'Y'
                """, (breed_code,))
                result = cursor.fetchone()
                if result:
                    return result[0]
                else:
                    return breed_code  # 코드를 찾을 수 없으면 원본 코드 반환
        except Error as e:
            print(f"❌ 견종 코드 '{breed_code}' 조회 실패: {e}")
            return breed_code
    
    def get_codes_by_group(self, group_cd: str) -> List[Dict[str, str]]:
        """그룹별 코드 목록 조회"""
        try:
            with self.db_connection.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT cd, cd_nm
                    FROM cmn_code
                    WHERE group_cd = %s AND use_yn = 'Y'
                    ORDER BY cd_nm
                """, (group_cd,))
                codes = cursor.fetchall()
                print(f"✅ {group_cd} 코드 {len(codes)}개 조회")
                return codes
        except Error as e:
            print(f"❌ {group_cd} 코드 조회 실패: {e}")
            return []
