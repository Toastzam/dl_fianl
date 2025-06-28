import React, { useState, useEffect } from 'react';

const SearchPetPage = ({ searchResults, onSelectDog, onBackToSearch, originalImage }) => {
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
    if (score >= 0.8) return { grade: 'A', color: '#4ECDC4', label: '높은 유사도' };
    if (score >= 0.7) return { grade: 'B', color: '#45B7D1', label: '보통 유사도' };
    if (score >= 0.6) return { grade: 'C', color: '#96CEB4', label: '낮은 유사도' };
    return { grade: 'D', color: '#FFEAA7', label: '매우 낮음' };
  };

  const formatImagePath = (path) => {
    if (!path) return '';
    if (path.startsWith('http')) return path;
    return `http://localhost:8001/api/image/${path}`;
  };

  if (!searchResults || searchResults.length === 0) {
    return (
      <div style={{
        minHeight: '100vh',
        backgroundColor: '#f8f9fa',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px'
      }}>
        <div style={{
          textAlign: 'center',
          backgroundColor: 'white',
          padding: '60px',
          borderRadius: '20px',
          boxShadow: '0 10px 30px rgba(0,0,0,0.1)'
        }}>
          <div style={{ fontSize: '64px', marginBottom: '20px' }}>🔍</div>
          <h2 style={{ color: '#333', marginBottom: '15px', fontSize: '24px' }}>검색 결과가 없습니다</h2>
          <p style={{ color: '#666', marginBottom: '30px', fontSize: '16px' }}>다른 이미지로 다시 검색해보세요.</p>
          <button 
            onClick={onBackToSearch}
            style={{
              backgroundColor: '#4ECDC4',
              color: 'white',
              border: 'none',
              padding: '15px 30px',
              borderRadius: '10px',
              fontSize: '16px',
              fontWeight: 'bold',
              cursor: 'pointer',
              transition: 'background-color 0.3s ease'
            }}
            onMouseOver={(e) => e.target.style.backgroundColor = '#45B7D1'}
            onMouseOut={(e) => e.target.style.backgroundColor = '#4ECDC4'}
          >
            새로운 검색
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#f8f9fa',
      padding: '20px'
    }}>
      {/* 헤더 */}
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        marginBottom: '30px'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '20px'
        }}>
          <button
            onClick={onBackToSearch}
            style={{
              backgroundColor: 'transparent',
              border: '2px solid #4ECDC4',
              color: '#4ECDC4',
              padding: '10px 20px',
              borderRadius: '10px',
              fontSize: '16px',
              fontWeight: 'bold',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'all 0.3s ease'
            }}
            onMouseOver={(e) => {
              e.target.style.backgroundColor = '#4ECDC4';
              e.target.style.color = 'white';
            }}
            onMouseOut={(e) => {
              e.target.style.backgroundColor = 'transparent';
              e.target.style.color = '#4ECDC4';
            }}
          >
            ← 새로운 검색
          </button>
          
          <h1 style={{
            textAlign: 'center',
            color: '#333',
            fontSize: '28px',
            fontWeight: 'bold',
            margin: 0
          }}>
            유사한 강아지 검색 결과
          </h1>
          
          <div style={{ width: '120px' }}></div>
        </div>

        {/* 원본 이미지와 검색 정보 */}
        {originalImage && (
          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '15px',
            boxShadow: '0 5px 15px rgba(0,0,0,0.1)',
            marginBottom: '30px',
            display: 'flex',
            alignItems: 'center',
            gap: '20px'
          }}>
            <img
              src={originalImage}
              alt="검색 이미지"
              style={{
                width: '100px',
                height: '100px',
                objectFit: 'cover',
                borderRadius: '10px'
              }}
            />
            <div>
              <h3 style={{ margin: '0 0 10px 0', color: '#333' }}>검색된 이미지</h3>
              <p style={{ margin: '0', color: '#666' }}>
                총 {searchResults.length}마리의 유사한 강아지를 찾았습니다
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
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <label style={{ fontWeight: 'bold', color: '#333' }}>정렬:</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              style={{
                padding: '8px 12px',
                borderRadius: '8px',
                border: '2px solid #e9ecef',
                backgroundColor: 'white',
                fontSize: '14px'
              }}
            >
              <option value="similarity">유사도순</option>
              <option value="confidence">신뢰도순</option>
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <label style={{ fontWeight: 'bold', color: '#333' }}>등급 필터:</label>
            <select
              value={filterGrade}
              onChange={(e) => setFilterGrade(e.target.value)}
              style={{
                padding: '8px 12px',
                borderRadius: '8px',
                border: '2px solid #e9ecef',
                backgroundColor: 'white',
                fontSize: '14px'
              }}
            >
              <option value="all">전체</option>
              <option value="s">S등급 (매우 유사)</option>
              <option value="a">A등급 (높은 유사도)</option>
              <option value="b">B등급 (보통 유사도)</option>
              <option value="c">C등급 (낮은 유사도)</option>
              <option value="d">D등급 (매우 낮음)</option>
            </select>
          </div>

          <div style={{ marginLeft: 'auto', color: '#666', fontSize: '14px' }}>
            {sortedResults.length}개 결과
          </div>
        </div>
      </div>

      {/* 검색 결과 갤러리 */}
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))',
        gap: '25px',
        padding: '0 10px'
      }}>
        {sortedResults.map((dog, index) => {
          const grade = getSimilarityGrade(dog.combined_similarity || dog.overall_similarity || 0);
          const similarity = (dog.combined_similarity || dog.overall_similarity || 0) * 100;
          
          return (
            <div
              key={index}
              onClick={() => onSelectDog(dog)}
              style={{
                backgroundColor: 'white',
                borderRadius: '20px',
                overflow: 'hidden',
                boxShadow: '0 8px 25px rgba(0,0,0,0.1)',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                position: 'relative'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.transform = 'translateY(-5px)';
                e.currentTarget.style.boxShadow = '0 15px 35px rgba(0,0,0,0.15)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 8px 25px rgba(0,0,0,0.1)';
              }}
            >
              {/* 순위 배지 */}
              <div style={{
                position: 'absolute',
                top: '15px',
                left: '15px',
                backgroundColor: index < 3 ? '#FFD700' : '#4ECDC4',
                color: index < 3 ? '#333' : 'white',
                padding: '8px 12px',
                borderRadius: '20px',
                fontSize: '14px',
                fontWeight: 'bold',
                zIndex: 10,
                boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
              }}>
                #{index + 1}
              </div>

              {/* 등급 배지 */}
              <div style={{
                position: 'absolute',
                top: '15px',
                right: '15px',
                backgroundColor: grade.color,
                color: 'white',
                padding: '8px 12px',
                borderRadius: '20px',
                fontSize: '14px',
                fontWeight: 'bold',
                zIndex: 10,
                boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
              }}>
                {grade.grade}
              </div>

              {/* 강아지 이미지 - 원본과 키포인트 나란히 */}
              <div style={{ 
                display: 'flex', 
                height: '200px',
                backgroundColor: '#f8f9fa'
              }}>
                {/* 원본 이미지 */}
                <div style={{ 
                  flex: 1, 
                  position: 'relative',
                  borderRight: '2px solid #fff'
                }}>
                  <div style={{
                    position: 'absolute',
                    top: '10px',
                    left: '10px',
                    backgroundColor: 'rgba(0,0,0,0.7)',
                    color: 'white',
                    padding: '4px 8px',
                    borderRadius: '5px',
                    fontSize: '12px',
                    fontWeight: 'bold',
                    zIndex: 5
                  }}>
                    원본
                  </div>
                  <img
                    src={formatImagePath(dog.image_path)}
                    alt={`유사한 강아지 ${index + 1} - 원본`}
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover'
                    }}
                    onError={(e) => {
                      e.target.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="150" height="200" viewBox="0 0 150 200"><rect width="150" height="200" fill="%23f8f9fa"/><text x="75" y="100" text-anchor="middle" fill="%236c757d" font-family="Arial, sans-serif" font-size="12">원본 이미지</text></svg>';
                    }}
                  />
                </div>

                {/* 키포인트 이미지 */}
                <div style={{ 
                  flex: 1, 
                  position: 'relative'
                }}>
                  <div style={{
                    position: 'absolute',
                    top: '10px',
                    left: '10px',
                    backgroundColor: 'rgba(255,107,107,0.9)',
                    color: 'white',
                    padding: '4px 8px',
                    borderRadius: '5px',
                    fontSize: '12px',
                    fontWeight: 'bold',
                    zIndex: 5
                  }}>
                    키포인트
                  </div>
                  {dog.keypoint_image_path ? (
                    <img
                      src={formatImagePath(dog.keypoint_image_path)}
                      alt={`유사한 강아지 ${index + 1} - 키포인트`}
                      style={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover'
                      }}
                      onError={(e) => {
                        e.target.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="150" height="200" viewBox="0 0 150 200"><rect width="150" height="200" fill="%23333"/><circle cx="75" cy="60" r="3" fill="red" fill-opacity="0.3"/><circle cx="65" cy="80" r="3" fill="red" fill-opacity="0.3"/><circle cx="85" cy="80" r="3" fill="red" fill-opacity="0.3"/><circle cx="75" cy="120" r="3" fill="red" fill-opacity="0.3"/><line x1="75" y1="60" x2="65" y2="80" stroke="yellow" stroke-width="2" stroke-opacity="0.3"/><line x1="75" y1="60" x2="85" y2="80" stroke="yellow" stroke-width="2" stroke-opacity="0.3"/><line x1="75" y1="60" x2="75" y2="120" stroke="yellow" stroke-width="2" stroke-opacity="0.3"/><text x="75" y="150" text-anchor="middle" fill="white" font-family="Arial, sans-serif" font-size="10">키포인트</text></svg>';
                      }}
                    />
                  ) : (
                    <div style={{
                      width: '100%',
                      height: '100%',
                      backgroundColor: '#333',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontSize: '12px'
                    }}>
                      키포인트 없음
                    </div>
                  )}
                </div>
              </div>

              {/* 정보 영역 */}
              <div style={{ padding: '20px' }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '15px'
                }}>
                  <h3 style={{
                    margin: 0,
                    color: '#333',
                    fontSize: '18px',
                    fontWeight: 'bold'
                  }}>
                    유사도 {similarity.toFixed(1)}%
                  </h3>
                  <span style={{
                    color: grade.color,
                    fontSize: '14px',
                    fontWeight: 'bold'
                  }}>
                    {grade.label}
                  </span>
                </div>

                {/* 상세 유사도 정보 */}
                <div style={{ marginBottom: '15px' }}>
                  {dog.simclr_similarity && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                      <span style={{ color: '#666', fontSize: '14px' }}>외형 유사도:</span>
                      <span style={{ color: '#333', fontSize: '14px', fontWeight: 'bold' }}>
                        {(dog.simclr_similarity * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}
                  {dog.keypoint_similarity && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                      <span style={{ color: '#666', fontSize: '14px' }}>자세 유사도:</span>
                      <span style={{ color: '#333', fontSize: '14px', fontWeight: 'bold' }}>
                        {(dog.keypoint_similarity * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}
                </div>

                {/* 클릭 안내 */}
                <div style={{
                  padding: '12px',
                  backgroundColor: '#f8f9fa',
                  borderRadius: '10px',
                  textAlign: 'center',
                  color: '#666',
                  fontSize: '14px'
                }}>
                  클릭하여 상세 분석 보기 →
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* 하단 여백 */}
      <div style={{ height: '50px' }}></div>
    </div>
  );
};

export default SearchPetPage;
