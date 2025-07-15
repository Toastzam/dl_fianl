import { useState, useEffect } from 'react';
import api from '@/api/api';

export default function SkinCheck() {
  // 채팅용 상태 관리
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  
  // 현재 입력 상태
  const [images, setImages] = useState([]);
  const [imageFiles, setImageFiles] = useState([]);
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [restoring, setRestoring] = useState(true);
  const [sessionHistory, setSessionHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(true);

  // 📌 세션 이력 로드 함수 추가
  const loadSessionHistory = async () => {
    try {
      setLoadingHistory(true);
      const response = await api.get('/aivet/sessions/all');
      
      if (response.data.success) {
        setSessionHistory(response.data.sessions);
        console.log(`${response.data.count}개 세션 이력 로드 완료`);
      }
    } catch (error) {
      console.error('세션 이력 로드 오류:', error);
    } finally {
      setLoadingHistory(false);
    }
  };

  // 📌 컴포넌트 마운트 시 이력 로드
  useEffect(() => {
    loadSessionHistory();
  }, []);

  // 📌 세션 클릭 시 해당 대화 로드
  const loadSession = async (sessionId) => {
    try {
      const response = await api.get(`/aivet/session/${sessionId}/messages`);
      
      if (response.data.success && response.data.messages.length > 0) {
        const restoredMessages = response.data.messages.map(msg => ({
          type: msg.senderType,
          content: msg.content,
          timestamp: new Date(msg.createdAt),
        }));
        
        setMessages(restoredMessages);
        setSessionId(sessionId);
        localStorage.setItem('aivet_session_id', sessionId);
        console.log(`세션 ${sessionId} 로드 완료`);
      }
    } catch (error) {
      console.error('세션 로드 오류:', error);
      alert('세션을 불러오는 중 오류가 발생했습니다.');
    }
  };

  // 📌 컴포넌트 마운트 시 이력 로드
  useEffect(() => {
    loadSessionHistory();
  }, []);

  // 📌 페이지 로드 시 기존 대화 복원
  useEffect(() => {
    const restoreConversation = async () => {
      try {
        const savedSessionId = localStorage.getItem('aivet_session_id');
        
        if (savedSessionId) {
          console.log('기존 세션 발견:', savedSessionId);
          
          const response = await api.get(
            `/aivet/session/${savedSessionId}/messages`
          );
          
          if (response.data.success && response.data.messages.length > 0) {
            const restoredMessages = response.data.messages.map(msg => ({
              type: msg.senderType,
              content: msg.content,
              timestamp: new Date(msg.createdAt),
            }));
            
            setMessages(restoredMessages);
            setSessionId(savedSessionId);
            console.log(`${restoredMessages.length}개 메시지 복원 완료`);
          } else {
            console.log('해당 세션에 메시지가 없음');
          }
        } else {
          console.log('기존 세션 없음, 새 대화 시작');
        }
      } catch (error) {
        console.error('대화 복원 오류:', error);
        localStorage.removeItem('aivet_session_id');
      } finally {
        setRestoring(false);
      }
    };

    restoreConversation();
  }, []);

  // 📌 새 상담 시작 함수
  const startNewConsultation = async () => {
    try {
      const response = await api.post('/aivet/session/new');
      
      if (response.data.success) {
        const newSessionId = response.data.newSessionId;
        setSessionId(newSessionId);
        localStorage.setItem('aivet_session_id', newSessionId);
        
        setMessages([]);
        setText('');
        setImages([]);
        setImageFiles([]);
        
        alert('새로운 상담이 시작되었습니다! 🆕');
        console.log('새 세션 생성:', newSessionId);
      }
    } catch (error) {
      console.error('새 상담 시작 오류:', error);
      alert('새 상담 시작 중 오류가 발생했습니다.');
    }
  };

  // 📌 다중 이미지 처리
  const handleImageChange = (e) => {
    const files = Array.from(e.target.files);
    
    if (files.length > 5) {
      alert('이미지는 최대 5개까지 업로드 가능합니다.');
      e.target.value = '';
      return;
    }

    const totalImages = images.length + files.length;
    if (totalImages > 5) {
      alert(`총 ${totalImages}개가 선택되었습니다. 최대 5개까지만 가능합니다.`);
      e.target.value = '';
      return;
    }

    const newImageUrls = files.map(file => URL.createObjectURL(file));
    
    setImages(prev => [...prev, ...newImageUrls]);
    setImageFiles(prev => [...prev, ...files]);
    
    e.target.value = '';
  };

  // 📌 개별 이미지 삭제
  const removeImage = (index) => {
    const newImages = images.filter((_, i) => i !== index);
    const newImageFiles = imageFiles.filter((_, i) => i !== index);
    
    setImages(newImages);
    setImageFiles(newImageFiles);
  };

  // 📌 진단 방식 선택 (hybrid or default)
  const [diagnosisMode, setDiagnosisMode] = useState('default');

  // 📌 메시지 전송
  const sendMessage = async () => {
    console.log('🚀 sendMessage 시작');
    console.log('📝 text:', text);
    console.log('📸 imageFiles:', imageFiles);
    console.log('🟡 diagnosisMode (진단 모드):', diagnosisMode);
    
    if (!text.trim()) {
      alert('증상 설명을 입력해주세요.');
      return;
    }
    console.log('✅ 유효성 검사 통과');
    
    const userMessage = {
      type: 'user',
      content: text,
      images: [...images],
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    setText('');
    setImages([]);
    setImageFiles([]);
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) fileInput.value = '';

    try {
      console.log('📦 FormData 생성 시작');
      const formData = new FormData();
      
      formData.append('text', userMessage.content);

      if (imageFiles.length > 0) {
        imageFiles.forEach((file, index) => {
          formData.append('images', file);
        });
      }

      // 진단 모드 추가
      formData.append('mode', diagnosisMode); // 'hybrid' 또는 'default'
      console.log('🟢 FormData에 추가된 mode:', diagnosisMode);

      for (let [key, value] of formData.entries()) {
        console.log(`  [FormData] ${key}:`, value);
      }

      if (sessionId) {
        formData.append('sessionId', sessionId);
      }
      
      const response = await api.post(
        '/aivet/diagnose',
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );

      if (response.data.success) {
        if (response.data.sessionId) {
          setSessionId(response.data.sessionId);
          localStorage.setItem('aivet_session_id', response.data.sessionId);
        }

        // 하이브리드 여부 및 EfficientNet 결과 명확히 출력
        let aiMessage = { type: 'ai', content: '', timestamp: new Date() };
        const diagnosis = response.data.diagnosis;
        if (diagnosis && diagnosis.hybrid) {
          aiMessage.content = `\n[하이브리드 진단 모드: ON]`;
          aiMessage.content += `\n\n[EfficientNet 예측 결과]`;
          if (diagnosis.efficientnet_results && diagnosis.efficientnet_results.length > 0) {
            diagnosis.efficientnet_results.forEach((r, i) => {
              aiMessage.content += `\n  - 이미지 ${i+1}: class=${r.class}, confidence=${r.confidence?.toFixed(3)}, probs=${JSON.stringify(r.probabilities)}`;
            });
          }
          aiMessage.content += `\n\n[Gemini 진단 결과]\n`;
          aiMessage.content += (diagnosis.gemini_result && diagnosis.gemini_result.final_response) ? diagnosis.gemini_result.final_response : JSON.stringify(diagnosis.gemini_result);
        } else {
          // 기존 방식
          aiMessage.content = typeof diagnosis === 'string' ? diagnosis : JSON.stringify(diagnosis);
        }
        setMessages(prev => [...prev, aiMessage]);
        loadSessionHistory(); // 새 세션 생성 시 이력 새로고침
      } else {
        alert('진단 중 오류가 발생했습니다: ' + response.data.error);
      }
    } catch (error) {
      console.error('API 호출 오류:', error);
      alert('서버 연결 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  // 대화 복원 중이면 로딩 표시
  if (restoring) {
    return (
      <div className="flex max-w-7xl mx-auto gap-4 p-4">
        {/* 왼쪽 사이드바 (로딩 중) */}
        <div className="w-64 bg-blue-50 rounded-lg shadow-sm border border-blue-100 p-4">
          <div className="animate-pulse">
            <div className="h-4 bg-blue-200 rounded mb-2"></div>
            <div className="h-4 bg-blue-200 rounded"></div>
          </div>
        </div>
        
        {/* 메인 채팅 영역 (로딩 중) */}
        <div className="flex-1 flex flex-col h-[75vh]">
          <div className="bg-amber-300 text-black p-4 rounded-t-lg">
          </div>
          <div className="flex-1 flex items-center justify-center bg-white">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-amber-600 mx-auto mb-4"></div>
              <p className="text-gray-600">이전 대화를 불러오는 중...</p>
            </div>
          </div>
        </div>
        
        {/* 오른쪽 사이드바 (로딩 중) */}
        <div className="w-64 bg-blue-50 rounded-lg shadow-sm border border-blue-100 p-4">
          <div className="animate-pulse">
            <div className="h-4 bg-blue-200 rounded mb-2"></div>
            <div className="h-8 bg-blue-200 rounded mb-4"></div>
            <div className="h-4 bg-blue-200 rounded mb-2"></div>
            <div className="h-16 bg-blue-200 rounded"></div>
          </div>
        </div>
      </div>
    );
  }



  return (

    <div className="flex max-w-7xl mx-auto gap-4 p-4">
      {/* 왼쪽 사이드바 */}
      <div className="w-64 rounded-lg shadow-sm border border-gray-100 p-4 h-fit">
        {/* 소개글 섹션 */}
        <div className="mb-6">
          <h2 className="text-lg font-bold text-gray-800 mb-3">AI 피부 진단 서비스</h2>
          <div className="text-sm text-gray-700 space-y-2">
            <p>반려동물의 증상을 설명해주세요!</p>
            <p>이미지는 선택사항입니다 📸 (최대 5개)</p>
          </div>
        </div>

        {/* 참고 정보 섹션 */}
        <div>
          <h3 className="text-base font-semibold text-gray-800 mb-3">참고 정보</h3>
          <div className="space-y-3 text-sm">
            <div>
              <h4 className="font-medium text-blue-700 mb-1">자주 묻는 질문 (FAQ)</h4>
            </div>
            
            <div>
              <h4 className="font-medium text-blue-700 mb-1">피부 질환 예방 가이드</h4>
            </div>
            
            <div>
              <h4 className="font-medium text-blue-700 mb-1">AI 진단 주의사항</h4>
            </div>
          </div>
        </div>
      </div>

      {/* 메인 채팅 영역 */}
      <div className="flex-1 flex flex-col h-[75vh]">

        {/* 채팅 메시지 영역 */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-white relative">


          {messages.length === 0 ? (
            <div className="text-center text-gray-500 mt-8">
              <p className="text-lg">채팅을 시작해보세요!</p>
              <p className="text-sm mt-2">증상 설명과 함께 이미지를 첨부하면 더 정확한 진단이 가능합니다.</p>
            </div>
          ) : (
            messages.map((message, index) => (
              <div key={index} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                  message.type === 'user' 
                    ? 'bg-amber-200 text-black' 
                    : 'bg-white text-gray-800 border border-amber-300'
                }`}>
                  {message.images && message.images.length > 0 && (
                    <div className="grid grid-cols-2 gap-2 mb-2">
                      {message.images.map((img, imgIndex) => (
                        <img 
                          key={imgIndex}
                          src={img} 
                          alt={`업로드된 이미지 ${imgIndex + 1}`} 
                          className="w-full rounded-lg"
                        />
                      ))}
                    </div>
                  )}
                  <p className="whitespace-pre-wrap text-sm">{message.content}</p>
                  <p className={`text-xs mt-1 ${
                    message.type === 'user' ? 'text-amber-700' : 'text-gray-500'
                  }`}>
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))
          )}
          
          {loading && (
            <div className="flex justify-start">
              <div className="bg-white text-gray-800 border border-amber-300 max-w-xs px-4 py-2 rounded-lg">
                <p className="text-sm">🤖 AI가 진단 중입니다...</p>
              </div>
            </div>
          )}
        </div>

        {/* 입력 영역 */}
        <div className="bg-white p-4 flex-shrink-0 rounded-b-lg">
          {/* 진단 방식 선택 */}
          <div className="mb-2 flex gap-3 items-center">
            <span className="text-xs text-gray-600">진단 방식:</span>
            <button
              className={`px-3 py-1 rounded text-xs font-semibold border ${diagnosisMode === 'default' ? 'bg-amber-400 border-amber-400 text-black' : 'bg-white border-gray-300 text-gray-600'}`}
              onClick={() => setDiagnosisMode('default')}
              disabled={loading}
            >
              기존 방식
            </button>
            <button
              className={`px-3 py-1 rounded text-xs font-semibold border ${diagnosisMode === 'hybrid' ? 'bg-amber-400 border-amber-400 text-black' : 'bg-white border-gray-300 text-gray-600'}`}
              onClick={() => setDiagnosisMode('hybrid')}
              disabled={loading}
            >
              하이브리드
            </button>
          </div>
          {/* 다중 이미지 업로드 */}
          {/* <div className="flex items-center justify-between mb-3">
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={handleImageChange}
              className="flex-1 text-sm text-amber-800 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-amber-100 file:text-amber-800 hover:file:bg-amber-200"
            />
            <p className="text-xs text-amber-700 ml-3">
              {imageFiles.length > 0 ? `이미지 ${imageFiles.length}/5개` : '선택된 파일 없음'}
            </p>
            {images.length > 0 && (
              <div className="mt-3 grid grid-cols-5 gap-2">
                {images.map((img, index) => (
                  <div key={index} className="relative">
                    <img 
                      src={img} 
                      alt={`미리보기 ${index + 1}`} 
                      className="w-full h-16 object-cover rounded-lg"
                    />
                    <button
                      onClick={() => removeImage(index)}
                      className="absolute -top-1 -right-1 bg-red-500 text-white rounded-full w-5 h-5 text-xs flex items-center justify-center hover:bg-red-600"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div> */}
          <div className="flex items-center justify-between mb-3">
          {/* 왼쪽: 파일선택 + 개수 */}
          <div className="flex items-center gap-3">
            <label
              htmlFor="file-upload"
              className="cursor-pointer inline-block file:mr-0 py-2 px-4 rounded-lg border-0 text-sm font-semibold bg-amber-100 text-amber-800 hover:bg-amber-200"
              style={{ userSelect: 'none' }}
            >
              파일 선택
              <input
                id="file-upload"
                type="file"
                accept="image/*"
                multiple
                onChange={handleImageChange}
                className="hidden"
              />
            </label>
            <p className="text-xs text-amber-700 whitespace-nowrap">
              {imageFiles.length > 0 ? `이미지 ${imageFiles.length}/5개` : '선택된 파일 없음'}
            </p>
          </div>
          
          {/* 오른쪽: 이미지 미리보기 */}
          <div className="flex gap-2">
            {images.map((img, index) => (
              <div key={index} className="relative">
                <img src={img} alt={`미리보기 ${index + 1}`} className="w-12 h-12 object-cover rounded-lg" />
                <button onClick={() => removeImage(index)} className="absolute -top-1 -right-1 bg-red-500 text-white rounded-full w-4 h-4 text-xs flex items-center justify-center hover:bg-red-600">×</button>
              </div>
            ))}
          </div>
        </div>
          
          <div className="flex space-x-2">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="증상을 설명해주세요... (이미지는 선택사항)"
              className="flex-1 p-3 border border-amber-200 rounded-lg focus:ring-2 focus:ring-amber-300 focus:border-transparent resize-none"
              rows="2"
            />
            <button
              onClick={sendMessage}
              disabled={loading}
              className={`px-6 py-2 rounded-lg font-medium transition-colors ${
                loading 
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed' 
                  : 'bg-amber-400 text-black hover:bg-amber-500'
              }`}
            >
              {loading ? '전송 중...' : '전송'}
            </button>
          </div>
        </div>
      </div>

      {/* 오른쪽 사이드바 */}
      <div className="w-64 rounded-lg shadow-sm border border-gray-100 flex flex-col h-[75vh]">
        <div className="p-4 border-b flex-shrink-0">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">새로운 상담이 필요하신가요?</h3>
          <button
            onClick={startNewConsultation}
            className="w-full bg-blue-200 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
          >
            새 상담 시작
          </button>
        </div>

        <div className="p-4 border-b flex-1 flex flex-col min-h-0">
          <h3 className="text-lg font-semibold text-gray-800 mb-3 flex-shrink-0">최근 상담 이력</h3>
          <div className="flex-1 overflow-y-auto space-y-2 pr-2">
            {loadingHistory ? (
              <div className="text-center py-4">
                <div className="text-sm text-gray-500">이력 로딩 중...</div>
              </div>
            ) : sessionHistory.length > 0 ? (
              sessionHistory.map((session) => (
                <div 
                  key={session.sessionId}
                  className="p-3 bg-white rounded-lg border border-blue-50 cursor-pointer hover:bg-blue-50 transition-colors"
                  onClick={() => loadSession(session.sessionId)}
                >
                  <div className="text-sm font-medium text-gray-700">
                    {new Date(session.createdAt).toLocaleString('ko-KR', {
                      year: 'numeric',
                      month: '2-digit',
                      day: '2-digit',
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </div>
                  <div className="text-xs text-gray-500">
                    {session.summary || '메시지 없음'}
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-4">
                <div className="text-sm text-gray-500">상담 이력이 없습니다</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}