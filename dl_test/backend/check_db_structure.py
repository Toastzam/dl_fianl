"""
실제 데이터베이스 구조 확인 스크립트
"""

from app.database import DatabaseManager

def main():
    print("🔍 데이터베이스 구조 확인 중...")
    
    try:
        db = DatabaseManager()
        
        # 모든 테이블 목록 조회
        print("\n📋 데이터베이스의 모든 테이블:")
        tables = db.show_tables()
        
        if not tables:
            print("❌ 테이블이 없거나 조회할 수 없습니다.")
            return
        
        # 각 테이블의 구조 확인
        for table in tables:
            print(f"\n🔍 테이블 '{table}' 구조:")
            try:
                structure = db.describe_table(table)
                if structure:
                    print(f"  총 {len(structure)}개 컬럼:")
                    for col in structure:
                        key_info = ""
                        if col.get('Key') == 'PRI':
                            key_info = " (Primary Key)"
                        elif col.get('Key') == 'MUL':
                            key_info = " (Foreign Key)"
                        
                        null_info = "NOT NULL" if col.get('Null') == 'NO' else "NULL"
                        default_info = f", Default: {col.get('Default', 'None')}" if col.get('Default') else ""
                        
                        print(f"    - {col['Field']}: {col['Type']} {null_info}{default_info}{key_info}")
                else:
                    print("  구조 정보를 가져올 수 없습니다.")
            except Exception as e:
                print(f"  ❌ 테이블 '{table}' 구조 조회 실패: {e}")
        
        print(f"\n✅ 총 {len(tables)}개 테이블 확인 완료")
        
    except Exception as e:
        print(f"❌ 데이터베이스 연결 또는 조회 실패: {e}")

if __name__ == "__main__":
    main()
