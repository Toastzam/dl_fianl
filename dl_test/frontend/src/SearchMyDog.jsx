import React, { useState, useRef } from 'react';

const SearchMyDog = ({ onClose, onSearchResults }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [progress, setProgress] = useState(0);
  const progressRef = useRef(0);
  const fileInputRef = useRef(null);

  // API 서버 주소 설정 (다른 로컬에서 접속 가능)
  const getApiBaseUrl = () => {
    // 항상 서버의 실제 IP 사용 (환경변수, 프록시 무시)
    return 'http://192.168.0.46:8001';
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
    setProgress(0);
    progressRef.current = 0;
    // 가짜 진행률 애니메이션 시작
    // 약 9초에 95% 도달하도록: 1.8~2.1%씩 200ms마다 증가
    const progressInterval = setInterval(() => {
      progressRef.current = Math.min(progressRef.current + (Math.random() * 0.3 + 1.8), 95);
      setProgress(progressRef.current);
    }, 200);

    // 디버깅: API URL 확인
    const apiUrl = getApiBaseUrl();
    console.log('🔍 API URL:', apiUrl);
    console.log('🌐 현재 hostname:', window.location.hostname);
    console.log('📁 업로드할 파일:', selectedFile.name, selectedFile.type);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const fullUrl = `${apiUrl}/api/upload_and_search/`;
      console.log('📡 전체 요청 URL:', fullUrl);

      const response = await fetch(fullUrl, {
        method: 'POST',
        body: formData,
      });

      console.log('📬 응답 상태:', response.status, response.statusText);

      let data = null;
      try {
        data = await response.json();
      } catch (jsonErr) {
        setMessage('서버 응답을 해석할 수 없습니다. 관리자에게 문의하세요.');
        setLoading(false);
        return;
      }
      console.log('📦 응답 데이터:', data);

      // 강아지 이미지가 아닐 때(백엔드에서 판별 실패)
      if (!data.success && data.error === 'not_a_dog') {
        clearInterval(progressInterval);
        setProgress(100);
        let debugInfo = '';
        if (data.dog_check) {
          debugInfo = `\n(디버그: 키포인트 ${data.dog_check.num_keypoints}개, 평균신뢰도 ${data.dog_check.avg_score?.toFixed(2)}, SimCLR최대유사도 ${data.dog_check.max_simclr_similarity?.toFixed(3)})`;
        }
        setMessage((data.message || '강아지 이미지가 아닙니다.') + debugInfo);
        return;
      }

      if (data.success) {
        clearInterval(progressInterval);
        setProgress(100);
        console.log('🎯 검색 결과 개수:', data.results?.length || 0);
        console.log('🖼️  검색 결과 이미지 정보:');
        data.results?.forEach((dog, index) => {
          console.log(`  ${index + 1}. ID: ${dog.id}, 이름: ${dog.db_info?.name || dog.name || '이름없음'}`);
          console.log(`     이미지 URL: ${dog.image_url || dog.image_path}`);
          console.log(`     견종: ${dog.db_info?.breed_name || dog.db_info?.breed || dog.breed} (코드: ${dog.db_info?.breed_code || dog.breed_code})`);
          console.log(`     성별: ${dog.db_info?.gender || dog.gender} (코드: ${dog.db_info?.gender_code || dog.gender_code})`);
          console.log(`     입양상태: ${dog.db_info?.adoption_status || dog.adoption_status} (코드: ${dog.db_info?.adoption_status_code || dog.adoption_status_code})`);
          console.log(`     유사도: ${dog.combined_similarity || dog.overall_similarity}`);
        });

        console.log('📊 검색 메타데이터:', data.search_metadata);


        // 검색 결과, 원본 이미지, 키포인트 이미지, 메타데이터를 부모 컴포넌트로 전달
        // 반드시 /api/image/output_keypoints/ 경로로 접근해야 동적 생성/서빙이 보장됨
        let queryKeypointImageUrl = null;
        if (data.query_keypoint_image) {
          let filename = data.query_keypoint_image;
          if (filename.includes('/')) {
            filename = filename.split('/').pop();
          }
          queryKeypointImageUrl = `${getApiBaseUrl()}/api/image/output_keypoints/${filename}`;
        }

        // query_image가 로컬 경로(c:/.../uploads/...)면 /uploads/부터 잘라서 웹 경로로 변환
        let originalImageUrl = data.query_image;
        if (originalImageUrl && originalImageUrl.includes('/uploads/')) {
          const idx = originalImageUrl.lastIndexOf('/uploads/');
          if (idx !== -1) {
            originalImageUrl = originalImageUrl.slice(idx);
          }
        }
        onSearchResults(data.results, originalImageUrl, queryKeypointImageUrl, data.search_metadata);
        setMessage('검색 완료! 결과를 확인해보세요.');
      } else {
        clearInterval(progressInterval);
        setProgress(100);
        setMessage(data.message || '검색에 실패했습니다. 다시 시도해주세요.');
      }
    } catch (error) {
      clearInterval(progressInterval);
      setProgress(100);
      console.error('검색 오류:', error);
      setMessage('네트워크 오류가 발생했습니다. 다시 시도해주세요.');
    } finally {
      setLoading(false);
      setTimeout(() => setProgress(0), 800); // 완료 후 게이지 잠깐 보여주고 리셋
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
            보호견 유사도 검색
          </h1>
          <p style={{ margin: '10px 0 0 0', fontSize: '16px', opacity: 0.9 }}>
            AI 기반 유사도 분석
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
              잃어버린 강아지와 유사한 강아지를 찾아드립니다!
            </p>
            <p style={{ margin: '0', fontSize: '14px' }}>
              AI 분석으로 정확한 유사도를 계산합니다
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
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                <img
                  src="/upload_icon.png"
                  alt="업로드 아이콘"
                  style={{ width: '80px', height: '80px', marginBottom: '20px', display: 'block' }}
                />
                <p style={{ margin: '10px 0', fontSize: '18px', color: '#333', fontWeight: 'bold', textAlign: 'center' }}>
                  강아지 사진을 선택해주세요
                </p>
                <p style={{ margin: '0', fontSize: '14px', color: '#999', textAlign: 'center' }}>
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
          {/* 메시지 + 진행률 표시 */}
          {(message || loading) ? (
            <div
              style={{
                padding: '15px',
                borderRadius: '10px',
                fontSize: '14px',
                textAlign: 'center',
                backgroundColor: loading ? '#e3f2fd' : message.includes('완료') ? '#e8f5e8' : '#fff3e0',
                color: loading ? '#1976d2' : message.includes('완료') ? '#388e3c' : '#f57c00',
                border: `2px solid ${loading ? '#bbdefb' : message.includes('완료') ? '#c8e6c9' : '#ffcc02'}`,
                position: 'relative',
                marginBottom: '10px',
                minHeight: '38px',
              }}
            >
              {loading ? (
                <React.Fragment>
                  <span style={{ marginRight: '8px' }}>⏳</span>
                  <span style={{ fontWeight: 'bold' }}>{message}</span>
                  <div style={{
                    marginTop: '10px',
                    width: '100%',
                    height: '10px',
                    background: '#e0e0e0',
                    borderRadius: '5px',
                    overflow: 'hidden',
                  }}>
                    <div style={{
                      width: progress + '%',
                      height: '100%',
                      background: 'linear-gradient(90deg, #4ECDC4, #FF6B6B)',
                      transition: 'width 0.3s',
                    }} />
                  </div>
                  <div style={{ fontSize: '12px', color: '#888', marginTop: '2px' }}>{Math.floor(progress)}%</div>
                </React.Fragment>
              ) : message}
            </div>
          ) : null}

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
            {loading ? '🔍 분석 중...' : '내 강아지 찾기'}
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
