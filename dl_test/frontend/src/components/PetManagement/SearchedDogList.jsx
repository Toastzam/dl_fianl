import React, { useState, useEffect } from 'react';
import { AI_SEARCH_SERVER_URL } from '@/config/env';

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

// API 서버 주소 설정 (Vite 프록시 사용)
const getApiBaseUrl = () => {
  // 개발 환경에서는 Vite 프록시를 통해 AI 서버에 접근
  return '/ai-api';
};

const SearchedDogList = ({ searchResults, onSelectDog, onBackToSearch, originalImage, queryKeypointImage }) => {
  const [sortBy, setSortBy] = useState('similarity');
  const [filterGrade, setFilterGrade] = useState('all');
  const [sortedResults, setSortedResults] = useState([]);

  useEffect(() => {
    if (searchResults) {
      sortResults(searchResults);
    }
  }, [searchResults, sortBy, filterGrade]);

  const sortResults = (results) => {
    let filtered = results;
    
    // 등급 필터링
    if (filterGrade !== 'all') {
      filtered = results.filter(dog => {
        const grade = getSimilarityGrade(dog.combined_similarity || dog.overall_similarity || 0);
        return grade.grade.toLowerCase() === filterGrade;
      });
    }

    // 정렬
    const sorted = [...filtered].sort((a, b) => {
      if (sortBy === 'similarity') {
        return (b.combined_similarity || b.overall_similarity || 0) - (a.combined_similarity || a.overall_similarity || 0);
      } else {
        return (b.confidence || 0) - (a.confidence || 0);
      }
    });

    setSortedResults(sorted);
  };

  // 유사도 등급 계산
  const getSimilarityGrade = (score) => {
    if (score >= 0.9) return { grade: 'S', color: '#FF6B6B', label: '매우 유사' };
    if (score >= 0.8) return { grade: 'A', color: '#FF8E53', label: '매우 유사' };
    if (score >= 0.7) return { grade: 'B', color: '#FF6B9D', label: '유사' };
    if (score >= 0.6) return { grade: 'C', color: '#C44569', label: '조금 유사' };
    return { grade: 'D', color: '#786FA6', label: '약간 유사' };
  };

  // 검색 결과가 없는 경우
  if (!searchResults || searchResults.length === 0) {
    return (
      <div style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        minHeight: '100vh',
        padding: '20px',
        fontFamily: 'Arial, sans-serif'
      }}>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          backgroundColor: 'white',
          borderRadius: '20px',
          boxShadow: '0 10px 30px rgba(0,0,0,0.2)',
          overflow: 'hidden'
        }}>
          <div style={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white',
            padding: '30px',
            textAlign: 'center'
          }}>
            <h1 style={{ margin: '0 0 20px 0', fontSize: '32px', fontWeight: 'bold' }}>
              {'유사한 강아지 검색 결과'}
            </h1>
            <button
              onClick={onBackToSearch}
              style={{
                backgroundColor: 'rgba(255,255,255,0.2)',
                color: 'white',
                border: '2px solid white',
                padding: '12px 25px',
                borderRadius: '25px',
                fontSize: '16px',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
            >
              🔍 새로운 검색
            </button>
          </div>

          <div style={{
            textAlign: 'center',
            padding: '40px',
            color: '#999',
            fontSize: '18px'
          }}>
            유사한 강아지를 찾지 못했습니다 😥
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      minHeight: '100vh',
      padding: '20px',
      fontFamily: 'Arial, sans-serif'
    }}>
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        backgroundColor: 'white',
        borderRadius: '20px',
        boxShadow: '0 10px 30px rgba(0,0,0,0.2)',
        overflow: 'hidden'
      }}>
        {/* 헤더 */}
        <div style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          padding: '30px',
          textAlign: 'center'
        }}>
          <h1 style={{ margin: '0 0 20px 0', fontSize: '32px', fontWeight: 'bold' }}>
            {'유사한 강아지 검색 결과'}
          </h1>
          <p style={{ margin: '0 0 20px 0', fontSize: '18px', opacity: 0.9 }}>
            총 {searchResults?.length || 0}마리의 유사한 강아지를 찾았습니다
          </p>
          <button
            onClick={onBackToSearch}
            style={{
              backgroundColor: 'rgba(255,255,255,0.2)',
              color: 'white',
              border: '2px solid white',
              padding: '12px 25px',
              borderRadius: '25px',
              fontSize: '16px',
              cursor: 'pointer',
              transition: 'all 0.3s ease'
            }}
          >
            🔍 새로운 검색
          </button>
        </div>

        <div style={{ padding: '30px' }}>
          {/* 상단 이미지 비교 섹션 */}
          {originalImage && (
            <div style={{
              backgroundColor: 'white',
              padding: '25px',
              borderRadius: '15px',
              boxShadow: '0 5px 15px rgba(0,0,0,0.1)',
              marginBottom: '30px'
            }}>
              <h2 style={{
                fontSize: '20px',
                marginBottom: '20px',
                textAlign: 'center',
                color: '#333',
                fontWeight: 'bold'
              }}>
                검색된 이미지
              </h2>

              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '30px',
                alignItems: 'start'
              }}>
                {/* 원본 이미지 (업로드 후 백엔드 URL 사용) */}
                <div style={{ textAlign: 'center' }}>
                  <h3 style={{
                    fontSize: '16px',
                    marginBottom: '15px',
                    color: '#333',
                    fontWeight: 'bold'
                  }}>
                    원본 이미지
                  </h3>
                  <div style={{
                    borderRadius: '12px',
                    overflow: 'hidden',
                    boxShadow: '0 5px 15px rgba(0,0,0,0.2)',
                    minHeight: '300px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: '#f8f9fa'
                  }}>
                    {(() => {
                      // blob: URL이 들어오면 무시하고, /uploads/ 또는 /output_keypoints/ 등만 처리
                      const getOriginalImageUrl = (imgPath) => {
                        if (!imgPath) return '';
                        if (imgPath.startsWith('blob:')) {
                          // blob URL도 그대로 반환 (프론트에서 바로 표시)
                          return imgPath;
                        }
                        // 파일명만 추출해서 /image/uploads/로 요청 (경로에 관계없이)
                        const fileName = imgPath.split(/[\\/]/).pop();
                        return `${getApiBaseUrl()}/image/uploads/${fileName}`;
                      };
                      return (
                        <img
                          src={getOriginalImageUrl(originalImage)}
                          alt="검색한 강아지"
                          style={{
                            maxWidth: '100%',
                            maxHeight: '300px',
                            objectFit: 'contain',
                            borderRadius: '8px'
                          }}
                          onError={(e) => {
                            // 이미지가 진짜 없을 때만 placeholder로 대체
                            if (!e.target.dataset.error) {
                              e.target.dataset.error = '1';
                              e.target.src = '/placeholder-dog.jpg';
                            }
                          }}
                        />
                      );
                    })()}
                  </div>
                </div>

                {/* 키포인트 이미지 */}
                <div style={{ textAlign: 'center' }}>
                  <h3 style={{
                    fontSize: '16px',
                    marginBottom: '15px',
                    color: '#333',
                    fontWeight: 'bold'
                  }}>
                    키포인트
                  </h3>
                  <div style={{
                    borderRadius: '12px',
                    overflow: 'hidden',
                    boxShadow: '0 5px 15px rgba(0,0,0,0.2)',
                    minHeight: '300px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: '#f8f9fa'
                  }}>
                    {/* 키포인트 이미지도 백엔드 URL로 표시 (blob 대신) */}
                    {(() => {
                      const getQueryKeypointUrl = (imgPath) => {
                        if (!imgPath) return '';
                        const cleanPath = imgPath.startsWith('/') ? imgPath.slice(1) : imgPath;
                        // /image/로 시작하도록 수정 (api 제거)
                        return `${getApiBaseUrl()}/image/${cleanPath}`;
                      };
                      return queryKeypointImage ? (
                        <img
                          src={getQueryKeypointUrl(queryKeypointImage)}
                          alt="검색 이미지 키포인트"
                          style={{
                            maxWidth: '100%',
                            maxHeight: '300px',
                            objectFit: 'contain',
                            borderRadius: '8px'
                          }}
                          onError={(e) => {
                            console.log('검색 이미지 키포인트 로드 실패:', e.target.src);
                            e.target.src = 'data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"300\" height=\"300\" viewBox=\"0 0 300 300\"><rect width=\"300\" height=\"300\" fill=\"%23333\"/><circle cx=\"150\" cy=\"80\" r=\"6\" fill=\"red\" fill-opacity=\"0.8\"/><circle cx=\"120\" cy=\"110\" r=\"6\" fill=\"red\" fill-opacity=\"0.8\"/><circle cx=\"180\" cy=\"110\" r=\"6\" fill=\"red\" fill-opacity=\"0.8\"/><circle cx=\"150\" cy=\"180\" r=\"6\" fill=\"red\" fill-opacity=\"0.8\"/><circle cx=\"110\" cy=\"220\" r=\"6\" fill=\"red\" fill-opacity=\"0.8\"/><circle cx=\"190\" cy=\"220\" r=\"6\" fill=\"red\" fill-opacity=\"0.8\"/><line x1=\"150\" y1=\"80\" x2=\"120\" y2=\"110\" stroke=\"yellow\" stroke-width=\"3\" stroke-opacity=\"0.8\"/><line x1=\"150\" y1=\"80\" x2=\"180\" y2=\"110\" stroke=\"yellow\" stroke-width=\"3\" stroke-opacity=\"0.8\"/><line x1=\"150\" y1=\"80\" x2=\"150\" y2=\"180\" stroke=\"yellow\" stroke-width=\"3\" stroke-opacity=\"0.8\"/><line x1=\"150\" y1=\"180\" x2=\"110\" y2=\"220\" stroke=\"yellow\" stroke-width=\"3\" stroke-opacity=\"0.8\"/><line x1=\"150\" y1=\"180\" x2=\"190\" y2=\"220\" stroke=\"yellow\" stroke-width=\"3\" stroke-opacity=\"0.8\"/><text x=\"150\" y=\"270\" text-anchor=\"middle\" fill=\"white\" font-family=\"Arial\" font-size=\"16\">키포인트 분석</text></svg>';
                          }}
                        />
                      ) : (
                        <div style={{
                          color: '#666',
                          fontSize: '16px',
                          textAlign: 'center',
                          padding: '20px'
                        }}>
                          키포인트 이미지를 생성하는 중입니다...
                        </div>
                      );
                    })()}
                  </div>
                </div>
              </div>

              {/* 하단 정보 */}
              <div style={{
                marginTop: '20px',
                padding: '15px',
                background: '#f8f9fa',
                borderRadius: '12px',
                textAlign: 'center'
              }}>
                <p style={{
                  fontSize: '14px',
                  margin: 0,
                  color: '#666',
                  lineHeight: '1.5'
                }}>
                  총 {searchResults?.length || 0}마리의 유사한 강아지를 찾았습니다
                </p>
              </div>
            </div>
          )}

          {/* 필터 및 정렬 옵션 */}
          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '15px',
            boxShadow: '0 5px 15px rgba(0,0,0,0.1)',
            marginBottom: '30px',
            display: 'flex',
            gap: '20px',
            alignItems: 'center',
            flexWrap: 'wrap'
          }}>
            
            <div style={{ marginLeft: 'auto', color: '#666' }}>
              {sortedResults.length}마리 표시 중
            </div>
          </div>

          {/* 강아지 카드 목록 */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
            gap: '25px'
          }}>
            {sortedResults.slice(0, 6).map((dog, index) => {
              const similarityScore = dog.combined_similarity || dog.overall_similarity || 0;
              const grade = getSimilarityGrade(similarityScore);
              // 등급별 강조 스타일
              const gradeStyles = {
                S: {
                  border: '5px solid #FF6B6B',
                  boxShadow: '0 0 0 4px #FF6B6B33, 0 8px 25px rgba(0,0,0,0.18)'
                },
                A: {
                  border: '4px solid #FF8E53',
                  boxShadow: '0 0 0 3px #FF8E5333, 0 8px 25px rgba(0,0,0,0.15)'
                },
                B: {
                  border: '3px solid #FF6B9D',
                  boxShadow: '0 0 0 2px #FF6B9D33, 0 8px 25px rgba(0,0,0,0.13)'
                },
                C: {
                  border: '2px solid #C44569',
                  boxShadow: '0 0 0 1px #C4456933, 0 8px 25px rgba(0,0,0,0.10)'
                },
                D: {
                  border: '1.5px solid #786FA6',
                  boxShadow: '0 8px 25px rgba(0,0,0,0.08)'


                  
                }
              };
              const g = gradeStyles[grade.grade] || gradeStyles.D;
              return (
                <div
                  key={`${dog.id || index}-${dog.image_path}`}
                  onClick={() => onSelectDog(dog)}
                  style={{
                    backgroundColor: 'white',
                    borderRadius: '15px',
                    overflow: 'hidden',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    transform: 'translateY(0)',
                    border: g.border,
                    boxShadow: g.boxShadow
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-8px)';
                    e.currentTarget.style.boxShadow = '0 15px 35px rgba(0,0,0,0.25)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = g.boxShadow;
                  }}
                >
                  {/* 강아지 이미지 */}
                  <div style={{ position: 'relative', height: '200px', overflow: 'hidden' }}>
                    {/* 이미지 비교 컨테이너 */}
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 1fr',
                      height: '100%',
                      gap: '2px'
                    }}>
                      {/* 원본 이미지 */}
                      <div style={{ position: 'relative', overflow: 'hidden' }}>
                        {(() => {
                          const getDogImageUrl = (imgPath) => {
                            if (!imgPath) return '/placeholder-dog.jpg';
                            if (imgPath.startsWith('blob:')) return imgPath;
                            if (imgPath.startsWith('http://') || imgPath.startsWith('https://')) return imgPath;
                            const fileName = imgPath.split(/[\\/]/).pop();
                            return `${getApiBaseUrl()}/image/uploads/${fileName}`;
                          };
                          return (
                            <img
                              src={getDogImageUrl(dog.image_url || dog.image_path)}
                              alt={dog.name || '강아지'}
                              style={{
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover'
                              }}
                              onError={(e) => {
                                e.target.src = '/placeholder-dog.jpg';
                              }}
                            />
                          );
                        })()}
                        {/* 원본 라벨 */}
                        <div style={{
                          position: 'absolute',
                          bottom: '5px',
                          left: '5px',
                          backgroundColor: 'rgba(0,0,0,0.7)',
                          color: 'white',
                          padding: '2px 6px',
                          borderRadius: '8px',
                          fontSize: '10px',
                          fontWeight: 'bold'
                        }}>
                          원본
                        </div>
                      </div>

                      {/* 키포인트 이미지 */}
                      <div style={{ position: 'relative', overflow: 'hidden' }}>
                        {/* 키포인트 이미지 경로 앞 슬래시 제거 함수 */}
                        {(() => {
                      const getKeypointImageUrl = (keypointPath) => {
                        if (!keypointPath) return '';
                        const cleanPath = keypointPath.startsWith('/') ? keypointPath.slice(1) : keypointPath;
                        // /image/로 시작하도록 수정 (api 제거)
                        const url = `${getApiBaseUrl()}/image/${cleanPath}`;
                        console.log('[DogList] dog.keypoint_image_path:', keypointPath, '| 최종 URL:', url);
                        return url;
                      };
                          return (
                            <img
                              src={getKeypointImageUrl(dog.keypoint_image_path)}
                              alt="키포인트"
                              style={{
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover'
                              }}
                              onError={(e) => {
                                console.log('키포인트 이미지 로드 실패:', e.target.src);
                                e.target.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="150" height="150" viewBox="0 0 150 150"><rect width="150" height="150" fill="%23333"/><circle cx="75" cy="40" r="3" fill="red" fill-opacity="0.8"/><circle cx="60" cy="55" r="3" fill="red" fill-opacity="0.8"/><circle cx="90" cy="55" r="3" fill="red" fill-opacity="0.8"/><circle cx="75" cy="90" r="3" fill="red" fill-opacity="0.8"/><circle cx="55" cy="110" r="3" fill="red" fill-opacity="0.8"/><circle cx="95" cy="110" r="3" fill="red" fill-opacity="0.8"/><line x1="75" y1="40" x2="60" y2="55" stroke="yellow" stroke-width="2" stroke-opacity="0.8"/><line x1="75" y1="40" x2="90" y2="55" stroke="yellow" stroke-width="2" stroke-opacity="0.8"/><line x1="75" y1="40" x2="75" y2="90" stroke="yellow" stroke-width="2" stroke-opacity="0.8"/><line x1="75" y1="90" x2="55" y2="110" stroke="yellow" stroke-width="2" stroke-opacity="0.8"/><line x1="75" y1="90" x2="95" y2="110" stroke="yellow" stroke-width="2" stroke-opacity="0.8"/><text x="75" y="135" text-anchor="middle" fill="white" font-family="Arial" font-size="10">키포인트</text></svg>';
                              }}
                            />
                          );
                        })()}
                        {/* 키포인트 라벨 */}
                        <div style={{
                          position: 'absolute',
                          bottom: '5px',
                          right: '5px',
                          backgroundColor: 'rgba(0,0,0,0.7)',
                          color: 'white',
                          padding: '2px 6px',
                          borderRadius: '8px',
                          fontSize: '10px',
                          fontWeight: 'bold'
                        }}>
                          키포인트
                        </div>
                      </div>
                    </div>
                    
                    {/* 유사도 등급 배지 */}
                    <div style={{
                      position: 'absolute',
                      top: '10px',
                      right: '10px',
                      backgroundColor: grade.color,
                      color: 'white',
                      padding: '8px 12px',
                      borderRadius: '20px',
                      fontSize: '14px',
                      fontWeight: 'bold',
                      boxShadow: '0 3px 10px rgba(0,0,0,0.3)',
                      zIndex: 2
                    }}>
                      {grade.grade}등급
                    </div>

                    {/* 유사도 점수 */}
                    <div style={{
                      position: 'absolute',
                      top: '10px',
                      left: '10px',
                      backgroundColor: 'rgba(0,0,0,0.7)',
                      color: 'white',
                      padding: '6px 10px',
                      borderRadius: '15px',
                      fontSize: '12px',
                      fontWeight: 'bold'
                    }}>
                      {(similarityScore * 100).toFixed(1)}% 유사
                    </div>
                  </div>

                  {/* 강아지 정보 */}
                  <div style={{ padding: '20px' }}>
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'flex-start',
                      marginBottom: '15px'
                    }}>
                      <h3 style={{
                        margin: '0',
                        fontSize: '18px',
                        fontWeight: 'bold',
                        color: '#333'
                      }}>
                        {dog.db_info?.name || dog.name || '이름 없음'}
                      </h3>
                      <span style={{
                        backgroundColor: grade.color,
                        color: 'white',
                        padding: '4px 8px',
                        borderRadius: '12px',
                        fontSize: '11px',
                        fontWeight: 'bold'
                      }}>
                        {grade.label}
                      </span>
                    </div>

                      <div style={{ fontSize: '14px', color: '#666' }}>
                        <div style={{ marginBottom: '8px' }}>
                          <strong>품종:</strong> {dog.db_info?.breed_name || dog.db_info?.breed || dog.breed_name || dog.breed || '정보 없음'}
                        </div>
                        <div style={{ marginBottom: '8px' }}>
                          <strong>성별:</strong> {getGenderText(dog.db_info?.gender || dog.gender)}
                        </div>
                        <div style={{ marginBottom: '8px' }}>
                          <strong>무게:</strong> {dog.db_info?.weight ? `${dog.db_info.weight}kg` : (dog.weight ? `${dog.weight}kg` : '정보 없음')}
                        </div>
                        <div style={{ marginBottom: '8px' }}>
                          <strong>발견 위치:</strong> {dog.db_info?.location || dog.db_info?.found_location || dog.location || dog.found_location || '정보 없음'}
                        </div>
                        <div>
                          <strong>입양 상태:</strong> {getAdoptionStatusText(dog.db_info?.adoption_status || dog.adoption_status)}
                        </div>
                      </div>

                    {/* 유사도 상세 정보 */}
                    <div style={{
                      marginTop: '15px',
                      padding: '12px',
                      backgroundColor: '#f8f9fa',
                      borderRadius: '8px',
                      fontSize: '12px'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>✨ SimCLR 유사도:</span>
                        <span style={{ fontWeight: 'bold', color: grade.color }}>
                          {((dog.simclr_similarity || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '5px' }}>
                        <span>🦴 키포인트 유사도:</span>
                        <span style={{ fontWeight: 'bold', color: grade.color }}>
                          {((dog.keypoint_similarity || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '5px' }}>
                        <span>🎯 종합 유사도:</span>
                        <span style={{ fontWeight: 'bold', color: grade.color }}>
                          {(similarityScore * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {sortedResults.length === 0 && (
            <div style={{
              textAlign: 'center',
              padding: '40px',
              color: '#999',
              fontSize: '18px',
              backgroundColor: 'white',
              borderRadius: '15px',
              boxShadow: '0 5px 15px rgba(0,0,0,0.1)'
            }}>
              선택한 필터 조건에 맞는 강아지가 없습니다.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SearchedDogList;
