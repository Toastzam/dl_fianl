// 전화번호 변경 페이지
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { updatePhone, getMyInfo } from '@/api/api';

export default function EditInfo() {
  const [currentPhone, setCurrentPhone] = useState('');
  const [newPhone, setNewPhone] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    // 내 정보 불러오기 (현재 전화번호)
    getMyInfo().then(res => {
      setCurrentPhone(res.data.phone || '');
    }).catch(() => {
      setCurrentPhone('');
    });
  }, []);

  const handleChange = (e) => {
    setNewPhone(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!newPhone.trim()) {
      alert('새 전화번호를 입력하세요.');
      return;
    }
    setLoading(true);
    try {
      await updatePhone({ phone: newPhone });
      alert('전화번호가 성공적으로 변경되었습니다.');
      navigate('/');
    } catch (err) {
      alert('전화번호 변경에 실패했습니다.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-md mx-auto">
      <h1 className="text-2xl font-bold mb-4 text-center">전화번호 변경</h1>
      <div className="mb-4">
        <span className="font-semibold">현재 전화번호: </span>
        <span>{currentPhone ? currentPhone : '정보 없음'}</span>
      </div>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block mb-1 font-semibold">새 전화번호</label>
          <input
            type="tel"
            name="newPhone"
            value={newPhone}
            onChange={handleChange}
            className="w-full p-2 border rounded"
            placeholder="새 전화번호를 입력하세요"
            required
            disabled={loading}
          />
        </div>
        <button
          type="submit"
          className="w-full bg-blue-500 text-white py-2 rounded hover:bg-blue-600"
          disabled={loading}
        >
          {loading ? '변경 중...' : '전화번호 변경'}
        </button>
      </form>
    </div>
  );
}