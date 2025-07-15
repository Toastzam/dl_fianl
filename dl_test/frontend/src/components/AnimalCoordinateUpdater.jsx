import React, { useState, useEffect } from 'react';
import api from '@/api/api';

const KAKAO_MAP_API_KEY = '9a19ecb7e00a6db6be9d34fd5f9f3c09';

const buttonStyle = {
  padding: '10px 20px',
  marginRight: '10px',
  marginBottom: '10px',
  fontSize: '16px',
  fontWeight: '600',
  backgroundColor: '#007bff',
  color: 'white',
  border: 'none',
  borderRadius: '4px',
  cursor: 'pointer',
  boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
  transition: 'background-color 0.3s ease'
};

const disabledButtonStyle = {
  ...buttonStyle,
  backgroundColor: '#a0a0a0',
  cursor: 'not-allowed',
  boxShadow: 'none',
};

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const cleanPlace = (place) => {
  if (!place) return '';
  return place
    .replace(/(부근|주변|인근|앞|뒤|에서 인수|에서 회수|에서 구조|출동 인수|출동 구조)/g, '')
    .replace(/\s{2,}/g, ' ')
    .trim();
};

const AnimalCoordinateUpdater = () => {
  const [kakaoReady, setKakaoReady] = useState(false);
  const [convertedData, setConvertedData] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const script = document.createElement('script');
    script.src = `//dapi.kakao.com/v2/maps/sdk.js?appkey=${KAKAO_MAP_API_KEY}&autoload=false&libraries=services`;
    script.async = true;
    script.onload = () => {
      // SDK 로드 후 명시적 초기화
      window.kakao.maps.load(() => {
        console.log('✅ Kakao 지도 서비스 로딩 완료');
        setKakaoReady(true);
      });
    };
    document.head.appendChild(script);
  }, []);

  const fetchAnimals = async () => {
    setLoading(true);
    try {
      const res = await api.get('http://localhost:8080/api/animals/target-animals');
      const data = Array.isArray(res.data) ? res.data : [];
      console.log('📦 받아온 발견 장소 데이터:', data);
      return data;
    } catch (err) {
      console.error('❌ 원본 데이터 조회 실패:', err);
      return [];
    } finally {
      setLoading(false);
    }
  };

  const convertCoordinates = async (data) => {
    if (!kakaoReady || !window.kakao?.maps?.services) {
      alert('카카오 맵 API가 아직 준비되지 않았습니다.');
      return;
    }

    if (!data || data.length === 0) {
      alert('조회된 데이터가 없습니다.');
      return;
    }

    setLoading(true);
    const ps = new window.kakao.maps.services.Places();
    const results = [];

    for (const animal of data) {
      if (!animal.place) continue;

      const cleanedPlace = cleanPlace(animal.place);
      console.log(`🔍 장소 전처리: '${animal.place}' → '${cleanedPlace}'`);

      // eslint-disable-next-line no-await-in-loop
      const coords = await new Promise((resolve) => {
        ps.keywordSearch(cleanedPlace, (places, status) => {
          if (status === window.kakao.maps.services.Status.OK && places.length > 0) {
            resolve({
              ...animal,
              placeLatitude: parseFloat(places[0].y),
              placeLongitude: parseFloat(places[0].x),
              originalPlace: animal.place,
              cleanedPlace: cleanedPlace
            });
          } else {
            console.warn(`❌ 장소 변환 실패: '${animal.place}' (${cleanedPlace})`);
            resolve({
              ...animal,
              placeLatitude: 0,
              placeLongitude: 0,
              originalPlace: animal.place,
              cleanedPlace: cleanedPlace
            });
          }
        });
      });

      results.push(coords);
      await delay(100);
    }

    console.log('✅ 좌표 변환 결과 리스트:', results);
    setConvertedData(results);
    setLoading(false);
  };

  const handleConvertCoordinates = async () => {
    const data = await fetchAnimals();
    await convertCoordinates(data);
  };

  const insertCoordinates = async () => {
    if (convertedData.length === 0) {
      alert('변환된 좌표 데이터가 없습니다.');
      return;
    }
  
    setLoading(true);
    try {
      for (const data of convertedData) {
        // 필요한 값만 추출해서 전송
        const payload = {
          id: data.id,
          place_latitude: data.placeLatitude,
          place_longitude: data.placeLongitude
        };

        console.log("🚀 보내는 payload:", payload, JSON.stringify(payload));

        await api.post('http://localhost:8080/api/animals/save-coordinates', payload);
        console.log('💾 저장 성공:', data.originalPlace);
      }
      alert('좌표 저장 완료');
    } catch (err) {
      console.error('❌ 좌표 저장 실패:', err);
      alert('저장 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h2>발견 장소 좌표 변환 및 저장</h2>
      <div>
        <button
          style={loading ? disabledButtonStyle : buttonStyle}
          onClick={handleConvertCoordinates}
          disabled={loading}
        >
          원본 조회 + 좌표 변환
        </button>
        <button
          style={convertedData.length === 0 || loading ? disabledButtonStyle : buttonStyle}
          onClick={insertCoordinates}
          disabled={convertedData.length === 0 || loading}
        >
          변환된 좌표 저장
        </button>
      </div>

      {loading && <p>처리 중입니다... 잠시만 기다려 주세요.</p>}

      <h3>변환된 좌표 데이터 ({convertedData.length}건)</h3>
      <ul style={{ maxHeight: 200, overflowY: 'auto' }}>
        {convertedData.map((c, i) => (
          <li key={i}>
            {c.originalPlace} → {c.cleanedPlace} : 위도 {c.placeLatitude}, 경도 {c.placeLongitude}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default AnimalCoordinateUpdater;
