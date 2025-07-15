"""
데이터베이스 연결 설정 및 유틸리티
"""
import mysql.connector
from mysql.connector import Error
import os
from typing import Dict, Any
from datetime import datetime, date
from decimal import Decimal

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env'))

# MySQL 연결 설정
DB_CONFIG = {
    'host': os.getenv('MYSQL_HOST'),
    'port': int(os.getenv('MYSQL_PORT', 3370)),
    'user': os.getenv('MYSQL_USER'),
    'password': os.getenv('MYSQL_PASSWORD'),
    'database': os.getenv('MYSQL_DATABASE'),
    'charset': 'utf8mb4',
    'autocommit': True,
    'use_pure': True,
    'connection_timeout': 10,
    'raise_on_warnings': True,
    'use_unicode': True,
    'ssl_disabled': True
}

# 전역 연결 로그 플래그 (첫 연결 시에만 로그 출력)
_first_connection_logged = False

class DatabaseConnection:
    """데이터베이스 연결 관리자 (싱글톤)"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized') or not self._initialized:
            self.db_config = DB_CONFIG
            self._initialized = True
    
    def get_connection(self):
        """MySQL DB 연결 생성"""
        global _first_connection_logged
        
        try:
            db_config = self.db_config.copy()
            host_value = db_config.get('host')
            
            if not host_value:
                db_config['host'] = '127.0.0.1'
            elif str(host_value).strip() in ['localhost', '.', '::1']:
                db_config['host'] = '127.0.0.1'
            
            connection = mysql.connector.connect(**db_config)
            
            # 첫 번째 연결 성공 시에만 로그 출력
            if not _first_connection_logged:
                print(f"✅ MySQL 데이터베이스 연결 성공: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
                _first_connection_logged = True
                
            return connection
        except Error as e:
            print(f"❌ DB 연결 오류: {e}")
            print(f"[DEBUG] DB_CONFIG: {self.db_config}")
            raise

def serialize_datetime(obj):
    """datetime 객체를 JSON 직렬화 가능한 문자열로 변환"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return obj

def convert_datetime_to_string(data):
    """딕셔너리나 리스트의 datetime 객체들을 문자열로 변환"""
    if isinstance(data, dict):
        return {key: convert_datetime_to_string(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_datetime_to_string(item) for item in data]
    elif isinstance(data, (datetime, date)):
        return data.isoformat() if data else None
    elif isinstance(data, Decimal):
        return float(data) if data is not None else None
    else:
        return data

# 공용 데이터베이스 연결 인스턴스 (한 번만 생성)
_shared_db_connection = None

def get_shared_db_connection():
    """공용 데이터베이스 연결 인스턴스 반환"""
    global _shared_db_connection
    if _shared_db_connection is None:
        _shared_db_connection = DatabaseConnection()
    return _shared_db_connection
