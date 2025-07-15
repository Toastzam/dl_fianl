import React, { useEffect, useRef, useState } from 'react';
import SearchChatbotModal from './SearchChatbotModal';
import api from '@/api/api';

const KAKAO_MAP_API_KEY = '9a19ecb7e00a6db6be9d34fd5f9f3c09';

const shelterList = [
  '시흥동물누리보호센터', '양평군유기동물보호소', '화성동물보호센터',
  '광주TNR동물병원초월', '한국야생동물보호협회', '펫앤쉘터동물병원',
  '오산 유기동물보호소', '한국동물구조관리협회', '평택시유기동물보호소',
  '남양주시동물보호센터', '수원시 동물보호센터', '안성시 동물보호센터',
  '24시아이동물메디컬', '용인시 동물보호센터', '정샘동물병원',
  '부천시수의사회', '위더스 동물보호센터', '가평군유기동물보호소',
  '광주TNR동물병원송정', '가나동물병원', '구리반려동물문화센터', '광명반함센터',
];

const SearchResult = () => {
  const mapRef = useRef(null);
  const [map, setMap] = useState(null);
  const [coords, setCoords] = useState({ lat: 37.26644, lng: 127.000609 });
  const [address, setAddress] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [markers, setMarkers] = useState([]);
  const [circles, setCircles] = useState([]);
  const [searchedLocation, setSearchedLocation] = useState(null);
  const [showChatbot, setShowChatbot] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const predictionMarkerRef = useRef(null);
  const predictionCircleRef = useRef(null);

  useEffect(() => {
    const loadKakaoMapScript = () => {
      const existingScript = document.getElementById('kakao-map-script');
      if (existingScript) existingScript.remove();

      const script = document.createElement('script');
      script.id = 'kakao-map-script';
      script.src = `https://dapi.kakao.com/v2/maps/sdk.js?appkey=${KAKAO_MAP_API_KEY}&autoload=false&libraries=services,drawing`;
      script.async = true;

      script.onload = () => {
        window.kakao.maps.load(() => {
          initMap();
        });
      };

      document.head.appendChild(script);
    };

    const initMap = () => {
      const container = mapRef.current;
      const options = {
        center: new window.kakao.maps.LatLng(coords.lat, coords.lng),
        level: 3,
      };
      const mapInstance = new window.kakao.maps.Map(container, options);
      setMap(mapInstance);

      const marker = new window.kakao.maps.Marker({
        position: new window.kakao.maps.LatLng(coords.lat, coords.lng),
        map: mapInstance,
      });
      setMarkers([marker]);
    };

    loadKakaoMapScript();
  }, [coords]);

  const handleSearch = () => {
    if (!searchQuery.trim() || !map) return;

    const ps = new window.kakao.maps.services.Places();
    ps.keywordSearch(searchQuery, (data, status) => {
      if (status === window.kakao.maps.services.Status.OK) {
        markers.forEach(marker => marker.setMap(null));

        const place = data[0];
        const lat = parseFloat(place.y);
        const lng = parseFloat(place.x);
        const latLng = new window.kakao.maps.LatLng(place.y, place.x);

        map.setCenter(latLng);
        map.setLevel(5);

        const newMarker = new window.kakao.maps.Marker({
          map,
          position: latLng,
        });

        setMarkers([newMarker]);
        setAddress(place.address_name);
        setSearchedLocation({ lat: parseFloat(lat.toFixed(6)), lng: parseFloat(lng.toFixed(6)) });
      } else {
        alert('검색 결과가 없습니다.');
      }
    });
  };

  const handleClick = async () => {
    if (!searchQuery.trim()) {
      alert('먼저 보호소 이름을 검색하세요.');
      return;
    }

    try {
      const response = await api.post('/petcare/searchArea', { name: searchQuery });
      const locations = response.data.map(item => ({
        placeLat: item.place_latitude,
        placeLng: item.place_longitude,
      }));

      circles.forEach(circle => circle.setMap(null));

      const newCircles = locations.map(loc => {
        const position = new window.kakao.maps.LatLng(loc.placeLat, loc.placeLng);
        return new window.kakao.maps.Circle({
          map,
          center: position,
          radius: 300,
          strokeWeight: 3,
          strokeColor: '#FF0000',
          strokeOpacity: 0.3,
          strokeStyle: 'solid',
          fillColor: '#FF0000',
          fillOpacity: 0.15,
        });
      });

      setCircles(newCircles);
    } catch (error) {
      console.error('Spring boot 요청 실패:', error);
      alert('Spring boot 요청 중 오류 발생');
    }
  };

  const handlePredictionClick = async () => {
    if (!searchQuery.trim()) {
      alert('먼저 보호소 이름을 검색하세요.');
      return;
    }
  
    try {
      const response = await api.post('/petcare/predictArea', { name: searchQuery });
      const data = response.data;
  
      const predictionLat = data.center_latitude;
      const predictionLng = data.center_longitude;
      const radius = data.radius; // 단위: km 가 아니라 m 단위로 처리해야 함
  
      console.log({ lat: predictionLat, lng: predictionLng, radius });
  
      // 기존 예측 마커가 있다면 제거
      if (predictionMarkerRef.current) {
        predictionMarkerRef.current.setMap(null);
      }
  
      // 기존 원(circle)도 제거
      if (predictionCircleRef.current) {
        predictionCircleRef.current.setMap(null);
      }
  
      const redMarkerImageSrc = 'https://t1.daumcdn.net/localimg/localimages/07/mapapidoc/marker_red.png';
      const imageSize = new window.kakao.maps.Size(24, 35);
      const imageOption = { offset: new window.kakao.maps.Point(12, 35) };
      const redMarkerImage = new window.kakao.maps.MarkerImage(redMarkerImageSrc, imageSize, imageOption);
  
      const predictionMarker = new window.kakao.maps.Marker({
        position: new window.kakao.maps.LatLng(predictionLat, predictionLng),
        image: redMarkerImage,
        map: map,
        title: '예측 위치',
      });
  
      predictionMarkerRef.current = predictionMarker;
  
      // 반경 원(circle) 그리기
      const maxRadius = 50000; // 최대 반경 50km (미터 단위)
      const scaledRadius = Math.min(radius * 1000, maxRadius); // km → m, 상한 적용
      
      const predictionCircle = new window.kakao.maps.Circle({
        center: new window.kakao.maps.LatLng(predictionLat, predictionLng),
        radius: scaledRadius,
        strokeWeight: 2,
        strokeColor: '#0000FF',
        strokeOpacity: 0.8,
        strokeStyle: 'solid',
        fillColor: '#87CEFA',
        fillOpacity: 0.3,
      });
  
      predictionCircle.setMap(map);
      predictionCircleRef.current = predictionCircle;
  
      // 지도의 중심을 예측 위치로 이동 (선택)
      map.setCenter(new window.kakao.maps.LatLng(predictionLat, predictionLng));
  
    } catch (error) {
      console.error('Spring boot 요청 실패:', error);
      alert('Spring boot 요청 중 오류 발생');
    }
  };
  
  
  

  return (
    <>
      <div
        style={{
          width: '100%',
          maxWidth: '1300px',
          margin: '20px auto',
          padding: '0 20px',
          boxSizing: 'border-box',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
          backgroundColor: '#fff',
          borderRadius: 8,
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        }}
      >
        {/* 1행: 검색창 + 검색버튼 + 발견지역 분포 */}
        <div
          style={{
            display: 'flex',
            gap: '12px',
            justifyContent: 'center',
            alignItems: 'center',
            flexWrap: 'wrap',
          }}
        >
          {/* 검색창 */}
          <div style={{ position: 'relative', width: '300px' }}>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => {
                const input = e.target.value;
                setSearchQuery(input);
                if (input.trim() === '') {
                  setSuggestions([]);
                  setShowSuggestions(false);
                } else {
                  const filtered = shelterList.filter((shelter) =>
                    shelter.includes(input)
                  );
                  setSuggestions(filtered);
                  setShowSuggestions(true);
                }
              }}
              placeholder="보호소 이름 검색"
              style={{
                width: '100%',
                padding: 10,
                borderRadius: 6,
                border: '1px solid #ccc',
                fontSize: 16,
              }}
            />
            {showSuggestions && suggestions.length > 0 && (
              <ul
                style={{
                  position: 'absolute',
                  top: '100%',
                  left: 0,
                  right: 0,
                  backgroundColor: '#fff',
                  border: '1px solid #ccc',
                  borderTop: 'none',
                  maxHeight: 200,
                  overflowY: 'auto',
                  zIndex: 100,
                  margin: 0,
                  padding: 0,
                  listStyle: 'none',
                }}
              >
                {suggestions.map((shelter, idx) => (
                  <li
                    key={idx}
                    onClick={() => {
                      setSearchQuery(shelter);
                      setSuggestions([]);
                      setShowSuggestions(false);
                    }}
                    style={{
                      padding: '8px 12px',
                      cursor: 'pointer',
                      borderBottom: '1px solid #eee',
                    }}
                  >
                    {shelter}
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* 검색 버튼 */}
          <button
            onClick={handleSearch}
            style={{
              padding: '8px 24px',
              backgroundColor: '#007aff',
              color: '#fff',
              border: 'none',
              borderRadius: 6,
              fontSize: 16,
              fontWeight: '600',
              cursor: 'pointer',
              whiteSpace: 'nowrap',
            }}
          >
            검색
          </button>

          {/* 보호견 발견 지역 분포 */}
          <button
            onClick={handleClick}
            style={{
              padding: '8px 24px',
              backgroundColor: '#007aff',
              color: '#fff',
              border: 'none',
              borderRadius: 6,
              cursor: 'pointer',
              fontWeight: '600',
              whiteSpace: 'nowrap',
            }}
          >
            보호견 발견 지역 분포
          </button>
        </div>

        {/* 2행: 현재위치 + 예측 */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '16px',
            alignItems: 'center',
            flexWrap: 'wrap',
          }}
        >
          <div style={{ fontSize: 16, fontWeight: '500', color: '#333' }}>
            현재 위치: {address || '주소 불러오는 중...'}
          </div>

          <button
            onClick={handlePredictionClick}
            style={{
              padding: '8px 24px',
              backgroundColor: '#34c759',
              color: '#fff',
              border: 'none',
              borderRadius: 6,
              cursor: 'pointer',
              fontWeight: '600',
              whiteSpace: 'nowrap',
            }}
          >
            보호견 탐색 범위 예측
          </button>
        </div>
      </div>

      {/* 지도 */}
      <div
        ref={mapRef}
        style={{
          width: '1300px',
          height: '600px',
          margin: '40px auto 60px',
          borderRadius: 8,
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
        }}
      />

      {/* 챗봇 */}
      <button
        onClick={() => setShowChatbot(true)}
        style={{
          position: 'fixed',
          bottom: 20,
          right: 20,
          padding: '12px 18px',
          fontSize: 16,
          backgroundColor: '#007aff',
          color: 'white',
          border: 'none',
          borderRadius: '50%',
          cursor: 'pointer',
          boxShadow: '0 4px 8px rgba(0,0,0,0.3)',
          zIndex: 50,
        }}
      >
        💬
      </button>

      {showChatbot && <SearchChatbotModal onClose={() => setShowChatbot(false)} />}
    </>
  );
};

export default SearchResult;
