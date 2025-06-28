import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setResults(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('먼저 이미지를 선택해주세요.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8001/api/upload_and_search/', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResults(data);
      } else {
        setError(data.detail || '검색에 실패했습니다.');
      }
    } catch (err) {
      setError('서버 연결에 실패했습니다: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const getImageUrl = (imagePath) => {
    return `http://localhost:8001/api/image/${imagePath}`;
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🐕 강아지 유사도 검색 시스템</h1>
        <p>SimCLR + AP-10K 키포인트 기반 유사도 검색</p>
      </header>

      <main className="main-content">
        {/* 업로드 섹션 */}
        <div className="upload-section">
          <div className="upload-area">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="file-input"
              id="file-input"
            />
            <label htmlFor="file-input" className="file-label">
              📁 이미지 선택
            </label>
            
            {selectedFile && (
              <div className="selected-file">
                <p>선택된 파일: {selectedFile.name}</p>
                <img 
                  src={URL.createObjectURL(selectedFile)} 
                  alt="Selected" 
                  className="preview-image"
                />
              </div>
            )}
            
            <button 
              onClick={handleUpload} 
              disabled={!selectedFile || loading}
              className="upload-button"
            >
              {loading ? '🔍 검색 중...' : '🚀 유사도 검색 시작'}
            </button>
          </div>
        </div>

        {/* 에러 메시지 */}
        {error && (
          <div className="error-message">
            ❌ {error}
          </div>
        )}

        {/* 로딩 상태 */}
        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>강아지 유사도를 분석하고 있습니다...</p>
            <p>SimCLR 특징 추출 및 키포인트 검출 진행 중</p>
          </div>
        )}

        {/* 검색 결과 */}
        {results && (
          <div className="results-section">
            <h2>🎯 검색 결과</h2>
            
            {/* 쿼리 이미지 */}
            <div className="query-section">
              <h3>📷 입력 이미지</h3>
              <div className="image-comparison">
                <div className="image-item">
                  <h4>원본 이미지</h4>
                  <img 
                    src={getImageUrl(results.query_image)} 
                    alt="Query" 
                    className="result-image"
                    onError={(e) => {
                      console.error('이미지 로드 실패:', results.query_image);
                      e.target.style.display = 'none';
                    }}
                  />
                </div>
                <div className="image-item">
                  <h4>키포인트 검출 결과</h4>
                  {results.query_keypoint_image ? (
                    <img 
                      src={getImageUrl(results.query_keypoint_image)} 
                      alt="Query Keypoints" 
                      className="result-image"
                      onError={(e) => {
                        console.error('키포인트 이미지 로드 실패:', results.query_keypoint_image);
                        e.target.style.display = 'none';
                      }}
                    />
                  ) : (
                    <p>키포인트 검출 실패</p>
                  )}
                </div>
              </div>
            </div>

            {/* 유사 이미지 갤러리 */}
            <div className="gallery-section">
              <h3>🏆 유사한 강아지들 (복합 유사도 기준)</h3>
              <div className="results-gallery">
                {results.results.map((result, index) => (
                  <div key={index} className="result-card">
                    <div className="rank-badge">#{result.rank}</div>
                    
                    <div className="image-pair">
                      <div className="original-image">
                        <h4>원본</h4>
                        <img 
                          src={getImageUrl(result.image_path)} 
                          alt={`Similar ${index + 1}`}
                          className="gallery-image"
                          onError={(e) => {
                            console.error('원본 이미지 로드 실패:', result.image_path);
                            e.target.style.display = 'none';
                          }}
                        />
                      </div>
                      
                      <div className="keypoint-image">
                        <h4>키포인트</h4>
                        {result.keypoint_image_path ? (
                          <img 
                            src={getImageUrl(result.keypoint_image_path)} 
                            alt={`Keypoints ${index + 1}`}
                            className="gallery-image"
                            onError={(e) => {
                              console.error('키포인트 이미지 로드 실패:', result.keypoint_image_path);
                              e.target.style.display = 'none';
                            }}
                          />
                        ) : (
                          <p>키포인트 없음</p>
                        )}
                      </div>
                    </div>
                    
                    <div className="similarity-scores">
                      <div className="score-item">
                        <span className="score-label">🎨 SimCLR:</span>
                        <span className="score-value">{(result.simclr_similarity * 100).toFixed(1)}%</span>
                      </div>
                      <div className="score-item">
                        <span className="score-label">🦴 키포인트:</span>
                        <span className="score-value">{(result.keypoint_similarity * 100).toFixed(1)}%</span>
                      </div>
                      <div className="score-item combined">
                        <span className="score-label">🏆 복합 유사도:</span>
                        <span className="score-value">{(result.combined_similarity * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;