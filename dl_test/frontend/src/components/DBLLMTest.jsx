import React, { useState } from 'react';

export default function DBLLMTest() {
  const [query, setQuery] = useState('');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8080/petcare/AIprocess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const result = await response.json();
      setData(result);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">DB & AI Test Page</h1>
      <form onSubmit={handleSubmit} className="mb-4">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="질문을 입력하세요"
          className="border p-2 mr-2"
        />
        <button type="submit" className="bg-blue-500 text-white p-2 rounded">
          전송
        </button>
      </form>
      {loading ? (
        <p className="text-center">Loading...</p>
      ) : data ? (
        <div>
          <h2 className="text-xl font-semibold mb-2">Database Results</h2>
          <table className="min-w-full bg-white border border-gray-300 mb-4">
            <thead>
              <tr className="bg-gray-100">
                <th className="py-2 px-4 border-b">ID</th>
                <th className="py-2 px-4 border-b">시군명</th>
                <th className="py-2 px-4 border-b">발견장소</th>
              </tr>
            </thead>
            <tbody>
              {data.db_data.map((item) => (
                <tr key={item.id} className="hover:bg-gray-50">
                  <td className="py-2 px-4 border-b">{item.id}</td>
                  <td className="py-2 px-4 border-b">{item.sigunName}</td>
                  <td className="py-2 px-4 border-b">{item.foundPlace}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h2 className="text-xl font-semibold mb-2">AI Response</h2>
          <p className="bg-gray-100 p-4 rounded">
            {data.llm_result.error ? (
              <span className="text-red-500">{data.llm_result.error}</span>
            ) : (
              data.llm_result.response
            )}
          </p>
        </div>
      ) : (
        <p className="text-center">데이터를 불러오려면 질문을 제출하세요.</p>
      )}
    </div>
  );
}