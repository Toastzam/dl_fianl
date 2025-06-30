import React from 'react';

// 입양상태 변환 함수
const getAdoptionStatusText = (status) => {
  if (!status) return '정보 없음';
  
  const statusMap = {
    'PREPARING': '입양준비중',
    'APPLY_AVAILABLE': '입양가능',
    'ADOPTED': '입양완료',
    'HOLD': '보류',
    'UNAVAILABLE': '입양불가'
  };
  
  return statusMap[status] || status;
};

// 성별 변환 함수
const getGenderText = (gender) => {
  if (!gender) return '정보 없음';
  
  const genderMap = {
    'M': '수컷',
    'F': '암컷',
    'Q': '알수없음'
  };
  
  return genderMap[gender] || gender;
};

const DogDetailView = ({ dogData, onBack, queryKeypointImage, searchMetadata }) => {
  // 디버깅 정보
  console.log('DogDetailView props:', { dogData, queryKeypointImage, searchMetadata });
  
  // API 서버 주소 설정 (다른 로컬에서 접속 가능)
  const getApiBaseUrl = () => {
    // 환경변수가 있으면 사용, 없으면 자동 선택
    if (import.meta.env.VITE_API_URL) {
      return import.meta.env.VITE_API_URL;
    }
    
    // 개발 환경에서는 프록시 사용 (CORS 우회)
    if (import.meta.env.DEV) {
      return '/ai-api'; // Vite 프록시 경로 사용
    }
    
    // localhost로 접속 중인지 확인
    const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    
    if (isLocalhost) {
      // localhost에서 실행 중이면 localhost 사용
      return 'http://localhost:8001';
    } else {
      // 다른 IP에서 접속 중이면 실제 IP 사용
      return 'http://192.168.0.46:8001';
    }
  };
  
  // 기본 더미 데이터
  const defaultDogData = {
    rank: 1,
    image_path: 'sample_dog.jpg',
    keypoint_image_path: 'sample_dog_keypoints.jpg',
    simclr_similarity: 0.892,
    keypoint_similarity: 0.734,
    combined_similarity: 0.845,
  };

  const currentDog = dogData || defaultDogData;

  // 유사도 등급 계산
  const getSimilarityGrade = (score) => {
    if (score >= 0.9) return { grade: 'S', color: '#FF6B6B', label: '매우 유사' };
    if (score >= 0.8) return { grade: 'A', color: '#4ECDC4', label: '높은 유사도' };
    if (score >= 0.7) return { grade: 'B', color: '#45B7D1', label: '보통 유사도' };
    if (score >= 0.6) return { grade: 'C', color: '#96CEB4', label: '낮은 유사도' };
    return { grade: 'D', color: '#FFEAA7', label: '매우 낮음' };
  };

  const gradeInfo = getSimilarityGrade(currentDog.combined_similarity);

  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#f8f9fa',
      padding: '20px',
      color: '#333'
    }}>
      {/* 헤더 */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        marginBottom: '30px',
        maxWidth: '1200px',
        margin: '0 auto 30px auto'
      }}>
        <button
          onClick={onBack}
          style={{
            background: '#4ECDC4',
            border: 'none',
            color: 'white',
            padding: '12px 24px',
            borderRadius: '25px',
            cursor: 'pointer',
            marginRight: '20px',
            fontSize: '16px',
            transition: 'all 0.3s ease'
          }}
        >
          ← 돌아가기
        </button>
        <h1 style={{
          fontSize: '32px',
          margin: 0,
          fontWeight: 'bold',
          color: '#333'
        }}>
          🐕 상세 분석 결과
        </h1>
      </div>

      <div style={{
        maxWidth: '1200px',
        margin: '0 auto'
      }}>
        {/* 키포인트 비교 섹션 */}
        <div style={{
          background: 'white',
          borderRadius: '20px',
          padding: '30px',
          marginBottom: '30px',
          boxShadow: '0 5px 15px rgba(0,0,0,0.1)',
          border: '1px solid #e9ecef'
        }}>
          <h2 style={{
            fontSize: '24px',
            marginBottom: '25px',
            textAlign: 'center',
            color: '#333',
            fontWeight: 'bold'
          }}>
            🔍 키포인트 비교 분석
          </h2>

          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '40px',
            alignItems: 'start'
          }}>
            {/* 검색 이미지의 키포인트 */}
            <div style={{ textAlign: 'center' }}>
              <h3 style={{
                fontSize: '20px',
                marginBottom: '15px',
                color: '#FFD93D',
                fontWeight: 'bold'
              }}>
                🐕 검색한 강아지의 키포인트
              </h3>
              <div style={{
                background: '#f8f9fa',
                borderRadius: '15px',
                padding: '15px',
                border: '2px solid #FFD93D',
                minHeight: '350px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                {queryKeypointImage ? (
                  <img
                    src={queryKeypointImage}
                    alt="검색 이미지 키포인트"
                    style={{
                      maxWidth: '100%',
                      maxHeight: '350px',
                      objectFit: 'contain',
                      borderRadius: '10px'
                    }}
                    onError={(e) => {
                      console.log('검색 이미지 키포인트 로드 실패:', e.target.src);
                      e.target.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300"><rect width="300" height="300" fill="%23333"/><circle cx="150" cy="80" r="6" fill="yellow" fill-opacity="0.3"/><circle cx="120" cy="110" r="6" fill="yellow" fill-opacity="0.3"/><circle cx="180" cy="110" r="6" fill="yellow" fill-opacity="0.3"/><circle cx="150" cy="180" r="6" fill="yellow" fill-opacity="0.3"/><circle cx="110" cy="220" r="6" fill="yellow" fill-opacity="0.3"/><circle cx="190" cy="220" r="6" fill="yellow" fill-opacity="0.3"/><line x1="150" y1="80" x2="120" y2="110" stroke="orange" stroke-width="3" stroke-opacity="0.3"/><line x1="150" y1="80" x2="180" y2="110" stroke="orange" stroke-width="3" stroke-opacity="0.3"/><line x1="150" y1="80" x2="150" y2="180" stroke="orange" stroke-width="3" stroke-opacity="0.3"/><line x1="150" y1="180" x2="110" y2="220" stroke="orange" stroke-width="3" stroke-opacity="0.3"/><line x1="150" y1="180" x2="190" y2="220" stroke="orange" stroke-width="3" stroke-opacity="0.3"/><text x="150" y="260" text-anchor="middle" fill="white" font-family="Arial" font-size="16">검색 키포인트</text></svg>';
                    }}
                  />
                ) : (
                  <div style={{
                    color: '#666',
                    fontSize: '16px'
                  }}>
                    키포인트 이미지 로딩 중...
                  </div>
                )}
              </div>
            </div>

            {/* 유사한 강아지의 키포인트 */}
            <div style={{ textAlign: 'center' }}>
              <h3 style={{
                fontSize: '20px',
                marginBottom: '15px',
                color: '#4ECDC4',
                fontWeight: 'bold'
              }}>
                🎯 유사한 강아지의 키포인트
              </h3>
              <div style={{
                background: '#f8f9fa',
                borderRadius: '15px',
                padding: '15px',
                border: '2px solid #4ECDC4',
                minHeight: '350px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <img
                  src={`${getApiBaseUrl()}/image/${currentDog.keypoint_image_path}`}
                  alt="유사한 강아지 키포인트"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '350px',
                    objectFit: 'contain',
                    borderRadius: '10px'
                  }}
                  onError={(e) => {
                    console.log('키포인트 이미지 로드 실패:', e.target.src);
                    e.target.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300"><rect width="300" height="300" fill="%23333"/><circle cx="150" cy="80" r="6" fill="red" fill-opacity="0.3"/><circle cx="120" cy="110" r="6" fill="red" fill-opacity="0.3"/><circle cx="180" cy="110" r="6" fill="red" fill-opacity="0.3"/><circle cx="150" cy="180" r="6" fill="red" fill-opacity="0.3"/><circle cx="110" cy="220" r="6" fill="red" fill-opacity="0.3"/><circle cx="190" cy="220" r="6" fill="red" fill-opacity="0.3"/><line x1="150" y1="80" x2="120" y2="110" stroke="yellow" stroke-width="3" stroke-opacity="0.3"/><line x1="150" y1="80" x2="180" y2="110" stroke="yellow" stroke-width="3" stroke-opacity="0.3"/><line x1="150" y1="80" x2="150" y2="180" stroke="yellow" stroke-width="3" stroke-opacity="0.3"/><line x1="150" y1="180" x2="110" y2="220" stroke="yellow" stroke-width="3" stroke-opacity="0.3"/><line x1="150" y1="180" x2="190" y2="220" stroke="yellow" stroke-width="3" stroke-opacity="0.3"/><text x="150" y="260" text-anchor="middle" fill="white" font-family="Arial" font-size="16">키포인트 이미지</text></svg>';
                  }}
                />
              </div>
            </div>
          </div>

          {/* 비교 설명 */}
          <div style={{
            marginTop: '25px',
            padding: '20px',
            background: '#f8f9fa',
            borderRadius: '15px',
            textAlign: 'center'
          }}>
            <p style={{
              fontSize: '16px',
              margin: 0,
              color: '#666',
              lineHeight: '1.5'
            }}>
              💡 <strong>키포인트 비교:</strong> 두 강아지의 주요 관절과 신체 부위의 위치를 비교하여 
              포즈와 자세의 유사성을 분석합니다. 비슷한 위치의 키포인트들이 높은 유사도를 나타냅니다.
            </p>
          </div>
        </div>

        {/* 분석 결과 */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '30px'
        }}>
          {/* 유사도 점수 */}
          <div style={{
            background: 'white',
            borderRadius: '20px',
            padding: '25px',
            boxShadow: '0 5px 15px rgba(0,0,0,0.1)',
            border: '1px solid #e9ecef'
          }}>
            <h3 style={{
              fontSize: '20px',
              marginBottom: '20px',
              color: '#333',
              fontWeight: 'bold'
            }}>
              📊 유사도 분석
            </h3>

            <div style={{ marginBottom: '20px' }}>
              <div style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                marginBottom: '15px'
              }}>
                <div style={{
                  width: '80px',
                  height: '80px',
                  borderRadius: '50%',
                  background: gradeInfo.color,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '32px',
                  fontWeight: 'bold',
                  color: 'white',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
                }}>
                  {gradeInfo.grade}
                </div>
              </div>
              <p style={{
                textAlign: 'center',
                fontSize: '18px',
                color: gradeInfo.color,
                margin: 0,
                fontWeight: 'bold'
              }}>
                {gradeInfo.label}
              </p>
            </div>

            <div>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '12px',
                padding: '10px',
                background: 'rgba(255, 255, 255, 0.1)',
                borderRadius: '8px'
              }}>
                <span style={{ color: '#666' }}>전체 유사도:</span>
                <span style={{ 
                  fontWeight: 'bold', 
                  color: gradeInfo.color,
                  fontSize: '16px' 
                }}>
                  {(currentDog.combined_similarity * 100).toFixed(1)}%
                </span>
              </div>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '12px',
                padding: '10px',
                background: '#f8f9fa',
                borderRadius: '8px'
              }}>
                <span style={{ color: '#666' }}>외형 유사도:</span>
                <span style={{ 
                  fontWeight: 'bold', 
                  color: '#4ECDC4',
                  fontSize: '16px' 
                }}>
                  {(currentDog.simclr_similarity * 100).toFixed(1)}%
                </span>
              </div>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '10px',
                background: '#f8f9fa',
                borderRadius: '8px'
              }}>
                <span style={{ color: '#666' }}>포즈 유사도:</span>
                <span style={{ 
                  fontWeight: 'bold', 
                  color: '#FFD93D',
                  fontSize: '16px' 
                }}>
                  {(currentDog.keypoint_similarity * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>

          {/* 순위 및 기타 정보 */}
          <div style={{
            background: 'white',
            borderRadius: '20px',
            padding: '25px',
            boxShadow: '0 5px 15px rgba(0,0,0,0.1)',
            border: '1px solid #e9ecef'
          }}>
            <h3 style={{
              fontSize: '20px',
              marginBottom: '20px',
              color: '#333',
              fontWeight: 'bold'
            }}>
              🏆 검색 결과 정보
            </h3>

            <div style={{
              textAlign: 'center',
              marginBottom: '20px'
            }}>
              <div style={{
                fontSize: '48px',
                color: '#FFD93D',
                fontWeight: 'bold',
                marginBottom: '10px'
              }}>
                #{currentDog.rank}
              </div>
              <p style={{
                fontSize: '18px',
                color: '#666',
                margin: 0
              }}>
                검색 결과 순위
              </p>
            </div>

            <div style={{
              background: '#f8f9fa',
              borderRadius: '15px',
              padding: '20px'
            }}>
              <h4 style={{
                color: '#333',
                marginBottom: '15px',
                fontSize: '16px'
              }}>
                📋 분석 요약
              </h4>
              <ul style={{
                listStyle: 'none',
                padding: 0,
                margin: 0,
                color: '#666',
                fontSize: '14px',
                lineHeight: '1.6'
              }}>
                <li style={{ marginBottom: '8px' }}>
                  • 외형 특성 매칭: {(currentDog.simclr_similarity * 100).toFixed(1)}%
                </li>
                <li style={{ marginBottom: '8px' }}>
                  • 포즈 유사성: {(currentDog.keypoint_similarity * 100).toFixed(1)}%
                </li>
                <li style={{ marginBottom: '8px' }}>
                  • 종합 판정: {gradeInfo.label}
                </li>
                <li>
                  • 신뢰도: {gradeInfo.grade}등급
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* 추가 상세 정보 섹션 */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr 1fr',
          gap: '20px',
          marginTop: '30px'
        }}>
          {/* 기술적 분석 */}
          <div style={{
            background: 'white',
            borderRadius: '15px',
            padding: '20px',
            boxShadow: '0 5px 15px rgba(0,0,0,0.1)',
            border: '1px solid #e9ecef'
          }}>
            <h4 style={{
              fontSize: '16px',
              marginBottom: '15px',
              color: '#4ECDC4',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              🔬 기술적 분석
            </h4>
            <div style={{ fontSize: '13px', color: '#666', lineHeight: '1.5' }}>
              <div style={{ marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span>특징 벡터 차원:</span>
                <span style={{ fontWeight: 'bold' }}>
                  {searchMetadata?.feature_dimension || '2048'}D
                </span>
              </div>
              <div style={{ marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span>키포인트 개수:</span>
                <span style={{ fontWeight: 'bold' }}>17개</span>
              </div>
              <div style={{ marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span>분석 시간:</span>
                <span style={{ fontWeight: 'bold' }}>
                  {searchMetadata?.processing_time || '0.34'}초
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>모델 버전:</span>
                <span style={{ fontWeight: 'bold' }}>
                  {searchMetadata?.model_version || 'v2.1'}
                </span>
              </div>
            </div>
          </div>

          {/* 매칭 상세 정보 */}
          <div style={{
            background: 'white',
            borderRadius: '15px',
            padding: '20px',
            boxShadow: '0 5px 15px rgba(0,0,0,0.1)',
            border: '1px solid #e9ecef'
          }}>
            <h4 style={{
              fontSize: '16px',
              marginBottom: '15px',
              color: '#FF6B6B',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              🎯 매칭 상세 정보
            </h4>
            <div style={{ fontSize: '13px', color: '#666', lineHeight: '1.5' }}>
              <div style={{ marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span>데이터베이스 크기:</span>
                <span style={{ fontWeight: 'bold' }}>
                  {searchMetadata?.database_size?.toLocaleString() || '10,000'}장
                </span>
              </div>
              <div style={{ marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span>검색된 유사 이미지:</span>
                <span style={{ fontWeight: 'bold' }}>
                  {searchMetadata?.searched_results || '5'}장
                </span>
              </div>
              <div style={{ marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span>신뢰도 임계값:</span>
                <span style={{ fontWeight: 'bold' }}>
                  {searchMetadata?.confidence_threshold || '0.60'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>매칭 알고리즘:</span>
                <span style={{ fontWeight: 'bold', fontSize: '11px' }}>
                  {searchMetadata?.algorithm || 'Hybrid AI'}
                </span>
              </div>
            </div>
          </div>

          {/* 신체 특징 */}
          <div style={{
            background: 'white',
            borderRadius: '15px',
            padding: '20px',
            boxShadow: '0 5px 15px rgba(0,0,0,0.1)',
            border: '1px solid #e9ecef'
          }}>
            <h4 style={{
              fontSize: '16px',
              marginBottom: '15px',
              color: '#FFD93D',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              📏 강아지 정보
            </h4>
            <div style={{ fontSize: '13px', color: '#666', lineHeight: '1.5' }}>
              <div style={{ marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span>성별:</span>
                <span style={{ fontWeight: 'bold' }}>
                  {getGenderText(currentDog.gender_code || currentDog.gender) || '정보없음'}
                </span>
              </div>
              <div style={{ marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span>견종:</span>
                <span style={{ fontWeight: 'bold' }}>{currentDog.breed || '믹스견'}</span>
              </div>
              <div style={{ marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span>중성화:</span>
                <span style={{ fontWeight: 'bold' }}>
                  {currentDog.neutered === true ? '완료 ✅' : 
                   currentDog.neutered === false ? '미완료 ❌' : 
                   (currentDog.neutered || '정보없음')}
                </span>
              </div>
              <div style={{ marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span>체중:</span>
                <span style={{ fontWeight: 'bold' }}>
                  {currentDog.weight ? `${currentDog.weight}kg` : '정보없음'}
                </span>
              </div>
              <div style={{ marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span>색상:</span>
                <span style={{ fontWeight: 'bold' }}>{currentDog.color || '정보없음'}</span>
              </div>
              <div style={{ marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span>입양상태:</span>
                <span style={{ fontWeight: 'bold' }}>{getAdoptionStatusText(currentDog.adoption_status_code || currentDog.adoption_status) || '정보없음'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>특징:</span>
                <span style={{ fontWeight: 'bold', maxWidth: '120px', textAlign: 'right' }}>
                  {currentDog.description ? 
                    (currentDog.description.length > 15 ? 
                      `${currentDog.description.substring(0, 15)}...` : 
                      currentDog.description
                    ) : '정보없음'
                  }
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DogDetailView;
