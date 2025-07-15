import { useState, useEffect, useRef } from 'react';
import api from '../../api/api';
import PetDetailModal from './PetDetailModal';
import { useCommonCode } from '../../contexts/CommonCodeContext';

function ChatConsult() {
  // 공통코드 가져오기
  const { getBreedName, getShelterName } = useCommonCode();

  // 세션 및 채팅 상태 관리
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  
  // 현재 입력 상태
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [restoring, setRestoring] = useState(true);
  const [loadingSessions, setLoadingSessions] = useState(true);

  // 모달 상태 관리
  const [selectedPet, setSelectedPet] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // 스크롤 ref 추가
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);

  // 자동 스크롤 함수 - 컨테이너 내부에서만 스크롤
  const scrollToBottom = () => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  };

  // 메시지가 업데이트될 때마다 자동 스크롤
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 로딩 상태가 변경될 때도 스크롤 (봇 응답 중일 때)
  useEffect(() => {
    if (loading) {
      scrollToBottom();
    }
  }, [loading]);

  // 펫 상세보기 열기
  const openPetDetail = (pet) => {
    setSelectedPet(pet);
    setIsModalOpen(true);
  };

  // 모달 닫기
  const closePetDetail = () => {
    setSelectedPet(null);
    setIsModalOpen(false);
  };

  // 📌 세션 목록 로드 함수
  const loadSessions = async () => {
    try {
      setLoadingSessions(true);
      const response = await api.get('/adopt/sessions');
      if (response.data.success) {
        setSessions(response.data.sessions);
        
        // 활성 세션이 있으면 선택
        const activeSessions = response.data.sessions.filter(s => s.status === 'ACTIVE');
        if (activeSessions.length > 0 && !currentSessionId) {
          setCurrentSessionId(activeSessions[0].sessionId);
          loadSessionMessages(activeSessions[0].sessionId);
        }
      }
    } catch (error) {
      console.error('세션 목록 로드 오류:', error);
      if (error.response?.status === 401 || error.response?.status === 403) {
        localStorage.removeItem('token');
        alert('로그인이 필요합니다. 다시 로그인해 주세요.');
      }
    } finally {
      setLoadingSessions(false);
    }
  };

  // 📌 새 세션 생성 함수
  const createNewSession = async (initialMessage = '새로운 상담을 시작합니다.') => {
    try {
      setLoading(true);
      const response = await api.post('/adopt/sessions', {
        initialMessage: initialMessage
      });
      
      if (response.data.success) {
        const newSession = response.data.session;
        setCurrentSessionId(newSession.sessionId);
        setSessions(prev => [newSession, ...prev]);
        
        // 초기 메시지 설정
        setMessages([
          { 
            type: 'bot', 
            content: '안녕하세요! 보호동물 입양 상담 챗봇입니다. 무엇을 도와드릴까요?',
            timestamp: new Date()
          }
        ]);
        
        return newSession.sessionId;
      }
    } catch (error) {
      console.error('새 세션 생성 오류:', error);
      alert('새 세션을 생성하는 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  // 📌 세션 메시지 로드 함수
  const loadSessionMessages = async (sessionId) => {
    try {
      const response = await api.get(`/adopt/sessions/${sessionId}/messages`);
      if (response.data.success) {
        const sessionMessages = response.data.messages.map(msg => {
          const baseMessage = {
            type: msg.messageType === 'USER' ? 'user' : 'bot',
            content: msg.content,
            timestamp: new Date(msg.createdAt)
          };
          
          // pets_json 데이터가 있으면 파싱하여 추가
          if (msg.petsJson) {
            try {
              const pets = JSON.parse(msg.petsJson);
              if (Array.isArray(pets) && pets.length > 0) {
                baseMessage.pets = pets;
              }
            } catch (e) {
              console.error('pets_json 파싱 오류:', e);
            }
          }
          
          return baseMessage;
        });
        
        setMessages(sessionMessages);
      }
    } catch (error) {
      console.error('세션 메시지 로드 오류:', error);
    }
  };

  // 📌 세션 선택 함수
  const selectSession = (sessionId) => {
    console.log(`[ChatConsult] 세션 전환: ${currentSessionId} → ${sessionId}`);
    setCurrentSessionId(sessionId);
    loadSessionMessages(sessionId);
  };

  // 📌 컴포넌트 마운트 시 세션 로드
  useEffect(() => {
    loadSessions();
    setRestoring(false);
    
    // 세션이 없을 때 초기 메시지 표시
    if (!currentSessionId && messages.length === 0) {
      setMessages([
        { 
          type: 'bot', 
          content: '안녕하세요! 보호동물 입양 상담 챗봇입니다. 새로운 상담을 시작하시거나 기존 상담을 선택해 주세요.',
          timestamp: new Date()
        }
      ]);
    }
  }, []);

  // 📌 새 상담 시작 함수
  const startNewConsultation = async () => {
    try {
      await createNewSession('새로운 입양 상담을 시작합니다.');
      alert('새로운 상담이 시작되었습니다! 🆕');
    } catch (error) {
      console.error('새 상담 시작 오류:', error);
      alert('새 상담을 시작하는 중 오류가 발생했습니다.');
    }
  };

  // 📌 메시지 전송
  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    // 현재 세션이 없으면 새 세션 생성
    let sessionId = currentSessionId;
    if (!sessionId) {
      sessionId = await createNewSession();
      if (!sessionId) return; // 세션 생성 실패 시 중단
    }

    console.log(`[ChatConsult] 메시지 전송 - 현재 세션: ${sessionId}, 입력: ${input}`);

    const userMessage = {
      type: 'user',
      content: input,
      timestamp: new Date()
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    const currentInput = input;
    setInput('');
    setLoading(true);

    try {
      // 세션 기반 메시지 전송 API 호출
      console.log(`[ChatConsult] API 호출: /adopt/sessions/${sessionId}/messages`);
      const res = await api.post(`/adopt/sessions/${sessionId}/messages`, { 
        prompt: currentInput 
      });
      
      const answer = res.data.answer || '죄송해요. 다시 말씀해 주세요!';
      const pets = Array.isArray(res.data.pets) ? res.data.pets : [];
      
      // 답변 메시지 추가
      const botMessage = { 
        type: 'bot', 
        content: answer,
        timestamp: new Date()
      };
      
      // pets 정보가 있으면 메시지에 포함
      if (Array.isArray(pets) && pets.length > 0) {
        botMessage.pets = pets;
      }
      
      const updatedMessages = [...newMessages, botMessage];

      setMessages(updatedMessages);
    } catch (e) {
      console.error('챗봇 API 호출 오류:', e);
      let errorMessage = '서버 오류가 발생했습니다.';
      
      if (e.response?.status === 401 || e.response?.status === 403) {
        errorMessage = '로그인이 필요합니다. 다시 로그인해 주세요.';
        localStorage.removeItem('token');
      }
      
      setMessages([...newMessages, { 
        type: 'bot', 
        content: errorMessage,
        timestamp: new Date()
      }]);
    } finally {
      setLoading(false);
    }
  };

  // 대화 복원 중이면 로딩 표시
  if (restoring) {
    return (
      <div className="flex max-w-7xl mx-auto gap-4 p-4">
        {/* 왼쪽 사이드바 (로딩 중) */}
        <div className="w-64 bg-green-50 rounded-lg shadow-sm border border-green-100 p-4">
          <div className="animate-pulse">
            <div className="h-4 bg-green-200 rounded mb-2"></div>
            <div className="h-4 bg-green-200 rounded"></div>
          </div>
        </div>
        
        {/* 메인 채팅 영역 (로딩 중) */}
        <div className="flex-1 flex flex-col h-[85vh]">
          <div className="bg-green-300 text-black p-4 rounded-t-lg">
          </div>
          <div className="flex-1 flex items-center justify-center bg-white">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600 mx-auto mb-4"></div>
              <p className="text-gray-600">이전 대화를 불러오는 중...</p>
            </div>
          </div>
        </div>
        
        {/* 오른쪽 사이드바 (로딩 중) */}
        <div className="w-64 bg-green-50 rounded-lg shadow-sm border border-green-100 p-4">
          <div className="animate-pulse">
            <div className="h-4 bg-green-200 rounded mb-2"></div>
            <div className="h-8 bg-green-200 rounded mb-4"></div>
            <div className="h-4 bg-green-200 rounded mb-2"></div>
            <div className="h-16 bg-green-200 rounded"></div>
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
          <h2 className="text-lg font-bold text-gray-800 mb-3">🐕 입양 상담 서비스</h2>
          <div className="text-sm text-gray-700 space-y-2">
            <p>보호동물 입양에 대해 궁금한 것이 있으시면 언제든 물어보세요!</p>
            <p>AI가 맞춤형 보호동물을 추천해드립니다 💕</p>
          </div>
        </div>

        {/* 참고 정보 섹션 */}
        <div>
          <h3 className="text-base font-semibold text-gray-800 mb-3">참고 정보</h3>
          <div className="space-y-3 text-sm">
            <div>
              <h4 className="font-medium text-green-700 mb-1">입양 절차 안내</h4>
            </div>
            
            <div>
              <h4 className="font-medium text-green-700 mb-1">입양 후 관리 가이드</h4>
            </div>
            
            <div>
              <h4 className="font-medium text-green-700 mb-1">입양 상담 주의사항</h4>
            </div>
          </div>
        </div>
      </div>

      {/* 메인 채팅 영역 */}
      <div className="flex-1 flex flex-col h-[85vh]">
        {/* 채팅 메시지 영역 */}
        <div 
          ref={messagesContainerRef}
          className="flex-1 overflow-y-auto p-4 space-y-4 bg-green-50 relative"
        >
          {messages.length === 0 ? (
            <div className="text-center text-gray-500 mt-8">
              {currentSessionId ? (
                <>
                  <p className="text-lg">이 세션에서 대화를 시작해보세요!</p>
                  <p className="text-sm mt-2">어떤 보호동물을 찾고 계신지 알려주세요.</p>
                </>
              ) : (
                <>
                  <p className="text-lg">새로운 상담을 시작해보세요!</p>
                  <p className="text-sm mt-2">오른쪽 "새 상담 시작" 버튼을 클릭하거나<br/>아래에 메시지를 입력해주세요.</p>
                </>
              )}
            </div>
          ) : (
            messages.map((message, index) => (
              <div key={index} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'} flex-col gap-1`}>
                <div className={`${message.image || (message.pets && message.pets.length > 0) ? 'max-w-2xl lg:max-w-3xl' : 'max-w-lg lg:max-w-xl'} px-4 py-3 rounded-lg ${
                  message.type === 'user' 
                    ? 'bg-green-200 text-black' 
                    : 'bg-white text-gray-800 border border-green-300'
                } ${message.type === 'user' ? 'self-end' : 'self-start'}`}>
                  
                  {/* 기존 이미지 방식 (기존 메시지 호환성) */}
                  {message.image ? (
                    <div className="flex gap-4 items-center">
                      {/* 이미지 영역 */}
                      <div className="flex-shrink-0">
                        <img 
                          src={message.image} 
                          alt="추천 보호동물" 
                          className="w-40 h-40 object-cover rounded-lg cursor-pointer hover:opacity-80 transition-opacity shadow-md border border-gray-200"
                          onClick={() => message.petData && openPetDetail(message.petData)}
                          title="클릭하면 상세정보를 볼 수 있습니다"
                        />
                      </div>
                      
                      {/* 정보 영역 */}
                      <div className="flex-1 min-w-0 flex flex-col justify-center">
                        <div className="text-sm leading-relaxed text-gray-700">
                          {message.content.split('\n').map((line, idx) => (
                            <div key={idx} className="mb-1 whitespace-nowrap">
                              {line}
                            </div>
                          ))}
                        </div>
                        {message.petData && (
                          <p className="text-xs text-green-600 mt-3 cursor-pointer hover:underline inline-flex items-center gap-1" 
                             onClick={() => openPetDetail(message.petData)}>
                            📋 상세정보 보기
                          </p>
                        )}
                      </div>
                    </div>
                  ) : message.pets && message.pets.length > 0 ? (
                    /* 새로운 pets 배열 방식 (이전 세션 복원) */
                    <div>
                      <p className="whitespace-pre-wrap text-sm mb-4">{message.content}</p>
                      <div className="grid grid-cols-1 gap-4">
                        {message.pets.map((pet, petIndex) => {
                          let gender = '-';
                          if (pet.gender_cd === 'F') gender = '암컷';
                          else if (pet.gender_cd === 'M') gender = '수컷';
                          else if (pet.gender_cd === 'Q') gender = '미상';

                          return (
                            <div key={petIndex} className="relative flex gap-4 items-center bg-gray-50 p-3 rounded-lg border">
                              {/* 추천 코멘트 표시 */}
                              {pet.recommendation_comment && (
                                <div className="absolute -top-2 left-4 bg-green-500 text-white text-xs px-2 py-1 rounded-full shadow-md z-10">
                                  {pet.recommendation_comment}
                                </div>
                              )}
                              
                              <div className="flex-shrink-0">
                                <img 
                                  src={pet.public_url} 
                                  alt="추천 보호동물" 
                                  className="w-32 h-32 object-cover rounded-lg cursor-pointer hover:opacity-80 transition-opacity shadow-md border border-gray-200"
                                  onClick={() => openPetDetail(pet)}
                                  title="클릭하면 상세정보를 볼 수 있습니다"
                                />
                              </div>
                              
                              <div className="flex-1 min-w-0 flex flex-col justify-center">
                                <div className="text-sm leading-relaxed text-gray-700">
                                  <div className="mb-1">품종: {getBreedName(pet.breed_cd)}</div>
                                  <div className="mb-1">성별: {gender} | 나이: {pet.birth_yyyy_mm || '-'}</div>
                                  <div className="mb-1">체중: {pet.weight_kg || '-'}kg | 털색: {pet.color || '-'}</div>
                                  <div className="mb-1">보호소: {pet.shelter_name || '정보 없음'}</div>
                                </div>
                                <p className="text-xs text-green-600 mt-2 cursor-pointer hover:underline inline-flex items-center gap-1" 
                                   onClick={() => openPetDetail(pet)}>
                                  📋 상세정보 보기
                                </p>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ) : (
                    <p className="whitespace-pre-wrap text-sm">{message.content}</p>
                  )}
                </div>
                
                {/* 시간 표시 - 대화창 밖으로 이동 */}
                <p className={`text-xs px-2 ${
                  message.type === 'user' 
                    ? 'text-green-700 self-end' 
                    : 'text-gray-500 self-start'
                }`}>
                  {message.timestamp.toLocaleTimeString()}
                </p>
              </div>
            ))
          )}
          
          {loading && (
            <div className="flex justify-start">
              <div className="bg-white text-gray-800 border border-green-300 max-w-xs px-4 py-2 rounded-lg">
                <p className="text-sm">🤖 AI가 상담 중입니다...</p>
              </div>
            </div>
          )}
          
          {/* 자동 스크롤을 위한 참조 요소 */}
          <div ref={messagesEndRef} />
        </div>

        {/* 입력 영역 */}
        <div className="bg-white p-4 flex-shrink-0 rounded-b-lg">
          <div className="flex space-x-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="입양에 대해 궁금한 것을 물어보세요..."
              className="flex-1 p-3 border border-green-200 rounded-lg focus:ring-2 focus:ring-green-300 focus:border-transparent resize-none"
              rows="2"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
            />
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              className={`px-6 py-2 rounded-lg font-medium transition-colors ${
                loading || !input.trim()
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed' 
                  : 'bg-green-400 text-black hover:bg-green-500'
              }`}
            >
              {loading ? '전송 중...' : '전송'}
            </button>
          </div>
        </div>
      </div>

      {/* 오른쪽 사이드바 */}
      <div className="w-64 rounded-lg shadow-sm border border-gray-100 flex flex-col h-[85vh]">
        <div className="p-4 border-b flex-shrink-0">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">새로운 상담이 필요하신가요?</h3>
          <button
            onClick={startNewConsultation}
            className="w-full bg-green-500 text-white py-2 px-4 rounded-lg hover:bg-green-600 transition-colors text-sm font-medium"
          >
            새 상담 시작
          </button>
        </div>

        <div className="p-4 border-b flex-1 flex flex-col min-h-0">
          <h3 className="text-lg font-semibold text-gray-800 mb-3 flex-shrink-0">최근 상담 이력</h3>
          <div className="flex-1 overflow-y-auto space-y-2 pr-2">
            {loadingSessions ? (
              <div className="text-center py-4">
                <div className="text-sm text-gray-500">이력 로딩 중...</div>
              </div>
            ) : sessions.length > 0 ? (
              sessions.map((session, index) => (
                <div 
                  key={session.sessionId}
                  onClick={() => selectSession(session.sessionId)}
                  className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                    currentSessionId === session.sessionId 
                      ? 'bg-green-100 border-green-300' 
                      : 'bg-white border-green-50 hover:bg-green-50'
                  }`}
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
                    {session.status === 'ACTIVE' ? '진행 중' : '완료'}
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-4">
                <div className="text-sm text-gray-500">상담 이력이 없습니다</div>
                <div className="text-xs text-gray-400 mt-1">첫 상담을 시작해보세요!</div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 펫 상세정보 모달 */}
      <PetDetailModal 
        pet={selectedPet}
        isOpen={isModalOpen}
        onClose={closePetDetail}
      />
    </div>
  );
}

export default ChatConsult;