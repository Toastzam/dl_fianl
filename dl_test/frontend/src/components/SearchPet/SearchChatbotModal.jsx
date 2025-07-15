import React, { useState } from 'react';
import api from '@/api/api';

const shelterList = [
  '시흥동물누리보호센터',
  '양평군유기동물보호소',
  '화성동물보호센터',
  '광주TNR동물병원초월',
  '한국야생동물보호협회',
  '펫앤쉘터동물병원',
  '오산 유기동물보호소',
  '한국동물구조관리협회',
  '평택시유기동물보호소',
  '남양주시동물보호센터',
  '수원시 동물보호센터',
  '안성시 동물보호센터',
  '24시아이동물메디컬',
  '용인시 동물보호센터',
  '정샘동물병원',
  '부천시수의사회',
  '위더스 동물보호센터',
  '가평군유기동물보호소',
  '광주TNR동물병원송정',
  '가나동물병원',
  '구리반려동물문화센터',
  '광명반함센터',
];

const SearchChatbotModal = ({ onClose, initialQuery }) => {
  const [messages, setMessages] = useState([
    { from: 'bot', text: '안녕하세요! 무엇을 도와드릴까요?' },
  ]);
  const [userInput, setUserInput] = useState(initialQuery || '');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);

  // axios 기반 FastAPI에 메시지 보내고 답변 받기
  const fetchBotResponse = async (query) => {
    setLoading(true);
    try {
      // 보호소 키워드 체크 (필요시 활용)
      const matchingShelter = shelterList.find(shelter => query.includes(shelter));

      const response = await api.post('/petcare/chat', { query });
      // api.js에서 자동으로 Authorization 헤더를 추가하므로 별도 헤더 설정 불필요

      setSessionId(response.data.session_id);
      return response.data.response;
    } catch (error) {
      console.error(error);
      return '죄송해요, 답변을 가져오는데 문제가 생겼어요.';
    } finally {
      setLoading(false);
    }
  };

  const closeSession = async () => {
    try {
      const response = await api.post('/petcare/close_chatsession');
    } catch (error) {
      console.error('세션 종료 실패:', error);
    }
  };

  const handleSendMessage = async () => {
    if (!userInput.trim() || loading) return;

    const userMsg = { from: 'user', text: userInput };
    setMessages((prev) => [...prev, userMsg]);
    setUserInput('');

    const botPlaceholder = { from: 'bot', text: '답변을 불러오는 중입니다...' };
    setMessages((prev) => [...prev, botPlaceholder]);

    const botReply = await fetchBotResponse(userInput);

    setMessages((prev) => [
      ...prev.slice(0, -1),
      { from: 'bot', text: botReply },
    ]);
  };

  return (
    <div
      style={{
        position: 'fixed',
        bottom: '90px',
        right: '20px',
        width: '300px',
        height: '400px',
        backgroundColor: 'white',
        borderRadius: '12px',
        boxShadow: '0 4px 16px rgba(0,0,0,0.2)',
        zIndex: 1001,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          backgroundColor: '#007aff',
          color: 'white',
          padding: '12px',
          fontWeight: 'bold',
          textAlign: 'center',
          position: 'relative',
        }}
      >
        보호견 구조 도우미 챗봇
        <button
          onClick={() => {
            closeSession();
            onClose();
            setSessionId(null);
          }}
          style={{
            position: 'absolute',
            top: '6px',
            right: '10px',
            background: 'transparent',
            border: 'none',
            color: 'white',
            fontSize: '16px',
            cursor: 'pointer',
          }}
        >
          ✕
        </button>
      </div>

      <div
        style={{
          flex: 1,
          padding: '10px',
          overflowY: 'auto',
          fontSize: '14px',
        }}
      >
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              margin: '6px 0',
              textAlign: msg.from === 'user' ? 'right' : 'left',
            }}
          >
            <div
              style={{
                display: 'inline-block',
                backgroundColor: msg.from === 'user' ? '#e1f5fe' : '#f1f1f1',
                padding: '8px 12px',
                borderRadius: '10px',
                maxWidth: '80%',
                whiteSpace: 'pre-wrap',
              }}
            >
              {msg.text}
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', borderTop: '1px solid #ccc' }}>
        <input
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
          disabled={loading}
          style={{
            flex: 1,
            padding: '8px',
            border: 'none',
            outline: 'none',
          }}
          placeholder="메시지를 입력하세요"
        />
        <button
          onClick={handleSendMessage}
          disabled={loading}
          style={{
            backgroundColor: loading ? '#aaa' : '#007aff',
            color: 'white',
            border: 'none',
            padding: '8px 12px',
            cursor: loading ? 'default' : 'pointer',
          }}
        >
          {loading ? '...' : '전송'}
        </button>
      </div>
    </div>
  );
};

export default SearchChatbotModal;
