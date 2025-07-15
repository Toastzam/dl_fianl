import React, { useState } from 'react';
import MissingPetInfo from './MissingPetInfo';
import AISearch from './ShowMap';
import api from '@/api/api'; // axios 인스턴스 사용

const SearchPetPage = () => {
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [nearestShelter, setNearestShelter] = useState(null);
  const [formData, setFormData] = useState({
    breed: '',
    age: '',
    weight: '',
    missingDateTime: '',
    missingLocation: ''
  });

  const handleLocationSelect = (coords) => {
    setSelectedLocation(coords);
  };

  const handleFormDataChange = (data) => {
    setFormData(data);
  };

  const handleNearestShelterFound = (shelter) => {
    setNearestShelter(shelter);
  };

  const handleAIBtnClick = async () => {
    const { breed, age, weight, missingDateTime, missingLocation } = formData;

    if (!breed || !missingDateTime || !missingLocation || !selectedLocation) {
      alert('모든 정보를 입력해주세요. (위치 포함)');
      return;
    }

    if (!nearestShelter) {
      alert('5km 이내 보호소가 선택되어야 합니다.');
      return;
    }

    const confirmMessage = `제보 내용을 등록합니다.`;

    if (!window.confirm(confirmMessage)) {
      alert('등록이 취소되었습니다.');
      return;
    }

    console.log("입력된 보호견 정보:", formData);
    console.log("선택된 지도 위치 좌표:", selectedLocation);
    console.log("선택된 보호소:", nearestShelter);

    let dateTimeWithSeconds = missingDateTime;
    if (dateTimeWithSeconds.length === 16) {
      dateTimeWithSeconds += ':00';
    }

    const todayDate = new Date().toISOString().split('T')[0]; // YYYY-MM-DD

    const payload = {
      place: missingLocation,
      breed,
      age: `${age}(년생)`,
      weight: parseFloat(weight),
      shelter: nearestShelter.name,
      shelter_call_number: nearestShelter.phoneNumber,
      shelter_address: nearestShelter.address,
      wgs84_latitude: nearestShelter.latitude,
      wgs84_longitude: nearestShelter.longitude,
      place_latitude: parseFloat(selectedLocation.lat.toFixed(6)),
      place_longitude: parseFloat(selectedLocation.lng.toFixed(6)),
      find_datetime: dateTimeWithSeconds,
      pet_received_date: todayDate
    };

    try {
      const response = await api.post('/petcare/saveMissingPet', payload);
      alert(response.data);
    } catch (error) {
      console.error("서버 전송 실패:", error);
      alert('서버 전송 중 오류 발생');
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', padding: '20px', alignItems: 'center' }}>
      <div style={{ display: 'flex', flexDirection: 'row', width: '100%' }}>
        <div style={{ flex: 2 }}>
          <AISearch selectedLocation={selectedLocation} nearestShelter={nearestShelter} />
        </div>
        <div style={{ flex: 1, marginLeft: '20px' }}>
          <MissingPetInfo
            onLocationSelect={handleLocationSelect}
            onFormDataChange={handleFormDataChange}
            onNearestShelterFound={handleNearestShelterFound}
          />
        </div>
      </div>

      <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'center', width: '100%' }}>
        <button
          type="button"
          onClick={handleAIBtnClick}
          style={{ padding: '24px 60px', backgroundColor: '#007bff', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer' }}
        >
          발견 신고 등록
        </button>
      </div>
    </div>
  );
};

export default SearchPetPage;
