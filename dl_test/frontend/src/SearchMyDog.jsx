import React, { useState, useRef } from 'react';

const SearchMyDog = ({ onClose, onSearchResults }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const fileInputRef = useRef(null);

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

  // 파일 선택 처리
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setMessage('');
    } else {
      setMessage('이미지 파일을 선택해주세요.');
    }
  };

  // 파일 업로드 및 검색
  const handleSearch = async () => {
    if (!selectedFile) {
      setMessage('먼저 강아지 사진을 선택해주세요.');
      return;
    }

    setLoading(true);
    setMessage('강아지 유사도 분석 중...');

    // 디버깅: API URL 확인
    const apiUrl = getApiBaseUrl();
    console.log('🔍 API URL:', apiUrl);
    console.log('🌐 현재 hostname:', window.location.hostname);
    console.log('📁 업로드할 파일:', selectedFile.name, selectedFile.type);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const fullUrl = `${apiUrl}/upload_and_search/`;
      console.log('📡 전체 요청 URL:', fullUrl);

      const response = await fetch(fullUrl, {
        method: 'POST',
        body: formData,
      });

      console.log('📬 응답 상태:', response.status, response.statusText);
      
      if (!response.ok) {
        throw new Error(`서버 오류: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('📦 응답 데이터:', data);
      
      if (data.success) {
        console.log('🎯 검색 결과 개수:', data.results?.length || 0);
        console.log('🖼️  검색 결과 이미지 정보:');
        data.results?.forEach((dog, index) => {
          console.log(`  ${index + 1}. ID: ${dog.id}, 이름: ${dog.name || '이름없음'}`);
          console.log(`     이미지 URL: ${dog.image_url || dog.image_path}`);
          console.log(`     견종: ${dog.breed} (코드: ${dog.breed_code})`);
          console.log(`     성별: ${dog.gender} (코드: ${dog.gender_code})`);
          console.log(`     입양상태: ${dog.adoption_status} (코드: ${dog.adoption_status_code})`);
          console.log(`     유사도: ${dog.combined_similarity || dog.overall_similarity}`);
        });
        
        console.log('📊 검색 메타데이터:', data.search_metadata);
        
        // 검색 결과, 원본 이미지, 키포인트 이미지, 메타데이터를 부모 컴포넌트로 전달
        const queryKeypointImageUrl = data.query_keypoint_image 
          ? `${getApiBaseUrl()}/image/${data.query_keypoint_image}`
          : null;
        
        onSearchResults(data.results, previewUrl, queryKeypointImageUrl, data.search_metadata);
        setMessage('검색 완료! 결과를 확인해보세요.');
      } else {
        setMessage('검색에 실패했습니다. 다시 시도해주세요.');
      }
    } catch (error) {
      console.error('검색 오류:', error);
      setMessage('네트워크 오류가 발생했습니다. 다시 시도해주세요.');
    } finally {
      setLoading(false);
    }
  };

  // 파일 선택 버튼 클릭
  const handleFileButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div
      style={{
        width: '100%',
        maxWidth: '600px',
        margin: '0 auto',
        backgroundColor: 'white',
        borderRadius: '20px',
        boxShadow: '0 10px 30px rgba(0,0,0,0.1)',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        minHeight: '500px',
      }}
    >
        {/* 헤더 */}
        <div
          style={{
            background: 'linear-gradient(135deg, #FF6B6B, #4ECDC4)',
            color: 'white',
            padding: '25px',
            textAlign: 'center',
            position: 'relative',
          }}
        >
          <h1 style={{ margin: 0, fontSize: '24px', fontWeight: 'bold' }}>
            🐕 강아지 유사도 검색
          </h1>
          <p style={{ margin: '10px 0 0 0', fontSize: '16px', opacity: 0.9 }}>
            SimCLR + AP-10K 키포인트 기반 유사도 분석
          </p>
        </div>

        {/* 메인 컨텐츠 */}
        <div
          style={{
            flex: 1,
            padding: '40px',
            display: 'flex',
            flexDirection: 'column',
            gap: '25px',
          }}
        >
          {/* 안내 메시지 */}
          <div style={{ textAlign: 'center', color: '#666' }}>
            <p style={{ margin: '0 0 10px 0', fontSize: '18px', color: '#333' }}>
              🔍 찾고 싶은 강아지와 유사한 강아지들을 찾아드려요!
            </p>
            <p style={{ margin: '0', fontSize: '14px' }}>
              SimCLR + 키포인트 분석으로 정확한 유사도를 계산합니다
            </p>
          </div>

          {/* 파일 업로드 영역 */}
          <div
            style={{
              border: '3px dashed #ddd',
              borderRadius: '15px',
              padding: '40px',
              textAlign: 'center',
              backgroundColor: previewUrl ? '#f9f9f9' : '#fafafa',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              minHeight: '200px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            onClick={handleFileButtonClick}
            onMouseOver={(e) => {
              e.target.style.borderColor = '#4ECDC4';
              e.target.style.backgroundColor = '#f0fffe';
            }}
            onMouseOut={(e) => {
              e.target.style.borderColor = '#ddd';
              e.target.style.backgroundColor = previewUrl ? '#f9f9f9' : '#fafafa';
            }}
          >
            {previewUrl ? (
              <div>
                <img
                  src={previewUrl}
                  alt="미리보기"
                  style={{
                    maxWidth: '200px',
                    maxHeight: '200px',
                    borderRadius: '15px',
                    marginBottom: '15px',
                    boxShadow: '0 5px 15px rgba(0,0,0,0.1)',
                  }}
                />
                <p style={{ margin: '10px 0', fontSize: '16px', color: '#333', fontWeight: 'bold' }}>
                  {selectedFile?.name}
                </p>
                <p style={{ margin: '5px 0', fontSize: '14px', color: '#666' }}>
                  클릭해서 다른 이미지 선택
                </p>
              </div>
            ) : (
              <div>
                <div style={{ fontSize: '80px', marginBottom: '20px' }}>📷</div>
                <p style={{ margin: '10px 0', fontSize: '18px', color: '#333', fontWeight: 'bold' }}>
                  강아지 사진을 선택해주세요
                </p>
                <p style={{ margin: '0', fontSize: '14px', color: '#999' }}>
                  JPG, PNG 파일만 지원됩니다
                </p>
              </div>
            )}
          </div>

          {/* 숨겨진 파일 입력 */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />

          {/* 메시지 표시 */}
          {message && (
            <div
              style={{
                padding: '15px',
                borderRadius: '10px',
                fontSize: '14px',
                textAlign: 'center',
                backgroundColor: loading ? '#e3f2fd' : message.includes('완료') ? '#e8f5e8' : '#fff3e0',
                color: loading ? '#1976d2' : message.includes('완료') ? '#388e3c' : '#f57c00',
                border: `2px solid ${loading ? '#bbdefb' : message.includes('완료') ? '#c8e6c9' : '#ffcc02'}`,
              }}
            >
              {loading && <span style={{ marginRight: '8px' }}>⏳</span>}
              {message}
            </div>
          )}

          {/* 검색 버튼 */}
          <button
            onClick={handleSearch}
            disabled={!selectedFile || loading}
            style={{
              backgroundColor: !selectedFile || loading ? '#ccc' : '#FF6B6B',
              color: 'white',
              border: 'none',
              padding: '18px 30px',
              borderRadius: '12px',
              fontSize: '16px',
              fontWeight: 'bold',
              cursor: !selectedFile || loading ? 'not-allowed' : 'pointer',
              transition: 'all 0.3s ease',
              boxShadow: !selectedFile || loading ? 'none' : '0 5px 15px rgba(255, 107, 107, 0.3)',
              marginBottom: '15px'
            }}
            onMouseOver={(e) => {
              if (!loading && selectedFile) {
                e.target.style.backgroundColor = '#ff5252';
                e.target.style.transform = 'translateY(-2px)';
              }
            }}
            onMouseOut={(e) => {
              if (!loading && selectedFile) {
                e.target.style.backgroundColor = '#FF6B6B';
                e.target.style.transform = 'translateY(0)';
              }
            }}
          >
            {loading ? '🔍 분석 중...' : '🚀 유사한 강아지 찾기'}
          </button>

          {/* 기능 설명 */}
          <div style={{ 
            fontSize: '13px', 
            color: '#999', 
            textAlign: 'center',
            padding: '15px',
            backgroundColor: '#f8f9fa',
            borderRadius: '10px',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: '10px' }}>
              <span>✨ SimCLR: 시각적 유사도 70%</span>
              <span>🦴 키포인트: 포즈 유사도 30%</span>
            </div>
          </div>
        </div>
    </div>
  );
};

export default SearchMyDog;
