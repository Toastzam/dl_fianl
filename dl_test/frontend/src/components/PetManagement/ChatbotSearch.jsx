// 챗봇 AI 검색 페이지
import { useState } from 'react';

export default function ChatbotSearch() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    { from: 'bot', text: '안녕하세요! 보호견 관리 챗봇입니다. 무엇을 도와드릴까요?' }
  ]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { from: 'user', text: input };
    const botResponse = generateBotResponse(input);

    setMessages(prev => [...prev, userMessage, { from: 'bot', text: botResponse }]);
    setInput('');
  };

  // 임의 챗봇 응답 함수 (추후 AI 연결 가능)
  const generateBotResponse = (userInput) => {
    if (userInput.includes('입양')) {
      return '보호견 입양 절차는 보호소 방문 후 상담을 통해 진행됩니다.';
    } else if (userInput.includes('예방접종')) {
      return '보호견의 예방접종은 1년에 1회 필수입니다. 상세 일정은 보호소에 문의하세요.';
    } else if (userInput.includes('상태')) {
      return '현재 보호 중인 견들의 건강 상태는 모두 양호합니다.';
    } else {
      return '죄송합니다. 해당 질문에 대한 정보가 없습니다. 보호소에 직접 문의해 주세요.';
    }
  };

  return (
    <div className="p-6 max-w-md mx-auto">
      <h1 className="text-2xl font-bold mb-4 text-center">보호견 관리 챗봇</h1>
      <div className="border rounded p-4 h-80 overflow-y-auto bg-gray-50 mb-4">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`mb-2 ${msg.from === 'user' ? 'text-right' : 'text-left'}`}
          >
            <span
              className={`inline-block px-3 py-2 rounded ${
                msg.from === 'user' ? 'bg-blue-200' : 'bg-green-200'
              }`}
            >
              {msg.text}
            </span>
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="flex space-x-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="flex-grow p-2 border rounded"
          placeholder="보호견 관련 질문을 입력하세요"
        />
        <button
          type="submit"
          className="bg-blue-500 text-white px-4 rounded hover:bg-blue-600"
        >
          전송
        </button>
      </form>
    </div>
  );
}