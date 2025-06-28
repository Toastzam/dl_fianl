import React, { useState, useRef } from 'react';


const SearchChatbotModal = ({ onClose, onSearchResults }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const fileInputRef = useRef(null);

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

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch('http://localhost:8001/api/upload_and_search/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('검색 중 오류가 발생했습니다.');
      }

      const data = await response.json();
      
      if (data.success) {
        // 검색 결과와 원본 이미지를 부모 컴포넌트로 전달하고 SearchPetPage로 이동
        onSearchResults(data.results, previewUrl);
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
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        backgroundColor: '#f8f9fa',
        zIndex: 1001,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px',
      }}
    >
      <div
        style={{
          width: '100%',
          maxWidth: '600px',
          backgroundColor: 'white',
          borderRadius: '20px',
          boxShadow: '0 20px 60px rgba(0,0,0,0.15)',
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
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '15px',
        }}
      >
        {/* 안내 메시지 */}
        <div style={{ textAlign: 'center', color: '#666' }}>
          <p style={{ margin: '0 0 10px 0', fontSize: '14px' }}>
            🔍 찾고 싶은 강아지와 유사한 강아지들을 찾아드려요!
          </p>
          <p style={{ margin: '0', fontSize: '12px' }}>
            SimCLR + 키포인트 분석으로 정확한 유사도를 계산합니다
          </p>
        </div>

        {/* 파일 업로드 영역 */}
        <div
          style={{
            border: '2px dashed #ddd',
            borderRadius: '8px',
            padding: '20px',
            textAlign: 'center',
            backgroundColor: previewUrl ? '#f9f9f9' : '#fafafa',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
          }}
          onClick={handleFileButtonClick}
        >
          {previewUrl ? (
            <div>
              <img
                src={previewUrl}
                alt="미리보기"
                style={{
                  maxWidth: '120px',
                  maxHeight: '120px',
                  borderRadius: '8px',
                  marginBottom: '10px',
                }}
              />
              <p style={{ margin: '5px 0', fontSize: '12px', color: '#666' }}>
                클릭해서 다른 이미지 선택
              </p>
            </div>
          ) : (
            <div>
              <div style={{ fontSize: '40px', marginBottom: '10px' }}>📷</div>
              <p style={{ margin: '5px 0', fontSize: '14px', color: '#666' }}>
                강아지 사진을 선택해주세요
              </p>
              <p style={{ margin: '0', fontSize: '12px', color: '#999' }}>
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
              padding: '10px',
              borderRadius: '6px',
              fontSize: '12px',
              textAlign: 'center',
              backgroundColor: loading ? '#e3f2fd' : message.includes('완료') ? '#e8f5e8' : '#fff3e0',
              color: loading ? '#1976d2' : message.includes('완료') ? '#388e3c' : '#f57c00',
            }}
          >
            {loading && <span style={{ marginRight: '5px' }}>⏳</span>}
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
            padding: '12px 20px',
            borderRadius: '8px',
            fontSize: '14px',
            fontWeight: 'bold',
            cursor: !selectedFile || loading ? 'not-allowed' : 'pointer',
            transition: 'all 0.3s ease',
          }}
        >
          {loading ? '🔍 분석 중...' : '🚀 유사한 강아지 찾기'}
        </button>

        {/* 기능 설명 */}
        <div style={{ fontSize: '11px', color: '#999', textAlign: 'center' }}>
          <p style={{ margin: '5px 0' }}>✨ SimCLR: 시각적 유사도 70%</p>
          <p style={{ margin: '5px 0' }}>🦴 키포인트: 포즈 유사도 30%</p>
        </div>
      </div>
    </div>
  );
};

export default SearchChatbotModal;
