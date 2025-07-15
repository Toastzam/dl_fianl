import React, { useState, useEffect } from 'react';

export default function DBTest() {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch('http://localhost:8080/petcare/test')
      .then(response => response.json())
      .then(data => setData(data))
      .catch(error => console.error('Error fetching data:', error));
  }, []);

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">DB Test Page - _animal_protection_status 테이블의 id, 시군명, 발견장소 10개 결과</h1>
      {data.length > 0 ? (
        <table className="min-w-full bg-white border border-gray-300">
          <thead>
            <tr className="bg-gray-100">
              <th className="py-2 px-4 border-b">ID</th>
              <th className="py-2 px-4 border-b">시군명</th>
              <th className="py-2 px-4 border-b">발견장소</th>
            </tr>
          </thead>
          <tbody>
            {data.map(item => (
              <tr key={item.id} className="hover:bg-gray-50">
                <td className="py-2 px-4 border-b">{item.id}</td>
                <td className="py-2 px-4 border-b">{item.sigunName}</td>
                <td className="py-2 px-4 border-b">{item.foundPlace}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="text-center">Loading data...</p>
      )}
    </div>
  );
}