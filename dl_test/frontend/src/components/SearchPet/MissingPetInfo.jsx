import React, { useState, useEffect } from 'react';
import api from '@/api/api';

const MissingPetInfo = ({ onLocationSelect, onFormDataChange, onNearestShelterFound }) => {
  const [formData, setFormData] = useState({
    breed: '',
    age: '',
    weight: '',
    missingDateTime: '',
    missingLocation: ''
  });

  const [breeds, setBreeds] = useState([]);
  const [shelters, setShelters] = useState([]);
  const [nearestShelter, setNearestShelter] = useState(null);

  useEffect(() => {
    api.post('/petcare/shelterList')
      .then(response => {
        setShelters(response.data);

        const uniqueBreeds = Array.from(new Set(response.data.map(item => item.breed)));
        setBreeds(uniqueBreeds);
      })
      .catch(error => {
        console.error('보호소 정보 불러오기 실패:', error);
      });
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    const updatedData = { ...formData, [name]: value };
    setFormData(updatedData);
    onFormDataChange(updatedData);
  };

  const handleSearch = () => {
    if (!formData.missingLocation) {
      alert('주소를 입력해주세요.');
      return;
    }

    if (window.kakao && window.kakao.maps) {
      const geocoder = new window.kakao.maps.services.Geocoder();
      geocoder.addressSearch(formData.missingLocation, (result, status) => {
        if (status === window.kakao.maps.services.Status.OK) {
          const coords = {
            lat: parseFloat(result[0].y),
            lng: parseFloat(result[0].x)
          };

          // 거리 계산 함수 (Haversine)
          const getDistance = (lat1, lng1, lat2, lng2) => {
            const toRad = deg => deg * Math.PI / 180;
            const R = 6371;
            const dLat = toRad(lat2 - lat1);
            const dLng = toRad(lng2 - lng1);
            const a = Math.sin(dLat / 2) ** 2 +
                      Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
                      Math.sin(dLng / 2) ** 2;
            return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
          };

          // 전체 보호소 중에서 가장 가까운 보호소 찾기
          const sheltersInRange = shelters;  

          let nearest = sheltersInRange[0];
          let minDist = getDistance(coords.lat, coords.lng, nearest.wgs84_latitude, nearest.wgs84_longitude);

          for (let i = 1; i < sheltersInRange.length; i++) {
            const dist = getDistance(coords.lat, coords.lng, sheltersInRange[i].wgs84_latitude, sheltersInRange[i].wgs84_longitude);
            if (dist < minDist) {
              nearest = sheltersInRange[i];
              minDist = dist;
            }
          }

          const nearestInfo = {
            name: nearest.shelter || '이름 없음',
            latitude: nearest.wgs84_latitude,
            longitude: nearest.wgs84_longitude,
            phoneNumber: nearest.shelter_call_number || '정보 없음',
            address: nearest.shelter_address || '정보 없음'
          };

          setNearestShelter(nearestInfo);
          onNearestShelterFound && onNearestShelterFound(nearestInfo);

          onLocationSelect(coords);

        } else {
          alert('주소 검색에 실패했습니다. 도로명 주소를 다시 확인해주세요.');
        }
      });
    }
  };

  return (
    <div style={{ padding: '20px', flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <form style={{ width: '100%', maxWidth: '400px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <h2 style={{ textAlign: 'center' }}>발견한 보호견 정보 입력</h2>

        {/* 품종 선택 */}
        <div>
          <label>품종:</label><br />
          <select
            name="breed"
            value={formData.breed}
            onChange={handleChange}
            required
            style={{ width: '100%', height: '36px' }}
          >
            <option value="">품종 선택</option>
            {breeds.map((breed, idx) => (
              <option key={idx} value={breed}>{breed}</option>
            ))}
          </select>
        </div>

        {/* 출생년도 선택 */}
        <div>
          <label>출생년도:</label><br />
          <select
            name="age"
            value={formData.age}
            onChange={handleChange}
            required
            style={{ width: '100%', height: '36px' }}
          >
            <option value="">출생년도 선택</option>
            {Array.from({ length: 20 }, (_, i) => {
              const year = new Date().getFullYear() - i;
              return <option key={year} value={year.toString()}>{year}</option>;
            })}
          </select>
        </div>

        <div>
          <label>무게 (kg):</label><br />
          <input type="number" name="weight" value={formData.weight} onChange={handleChange} required style={{ width: '100%' }} />
        </div>

        <div>
          <label>발견 시간:</label><br />
          <input
            type="datetime-local"
            name="missingDateTime"
            value={formData.missingDateTime}
            onChange={handleChange}
            required
            max={new Date().toISOString().slice(0, 16)}
            style={{ width: '100%' }}
          />
        </div>

        <div>
          <label>발견 장소 (도로명 주소 입력):</label><br />
          <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
            <input
              type="text"
              name="missingLocation"
              value={formData.missingLocation}
              onChange={handleChange}
              required
              style={{ flex: 1, height: '36px' }}
            />
            <button
              type="button"
              onClick={handleSearch}
              style={{
                padding: '0 16px',
                backgroundColor: '#007aff',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                height: '36px',
                alignSelf: 'center'
              }}
            >
              검색
            </button>
          </div>
          <p style={{ fontSize: '12px', marginTop: '4px' }}>예시: 서울특별시 강남구 테헤란로 123</p>
        </div>

        {/* 보호소 정보 표시 */}
        <div style={{ border: '1px dashed #ccc', padding: '12px', borderRadius: '6px', background: '#fefefe' }}>
          <h4>가장 가까운 보호소 정보</h4>
          <p><strong>보호소명:</strong> <span>{nearestShelter ? nearestShelter.name : ''}</span></p>
          <p><strong>전화번호:</strong> <span>{nearestShelter ? nearestShelter.phoneNumber : ''}</span></p>
          <p><strong>주소:</strong> <span>{nearestShelter ? nearestShelter.address : ''}</span></p>
        </div>
      </form>
    </div>
  );
};

export default MissingPetInfo;
