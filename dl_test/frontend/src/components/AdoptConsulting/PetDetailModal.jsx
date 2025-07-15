import { useState, useEffect } from 'react';
import { useCommonCodes } from '@/contexts/CommonCodeContext';
import api from '@/api/api';

export default function PetDetailModal({ pet, isOpen, onClose }) {
  // 공통코드 가져오기
  const { codes, loading: codesLoading, error: codesError } = useCommonCodes();
  
  // 보호소 정보 상태
  const [shelters, setShelters] = useState([]);
  const [sheltersLoading, setSheltersLoading] = useState(true);

  // 보호소 목록 로드
  useEffect(() => {
    const loadShelters = async () => {
      try {
        setSheltersLoading(true);
        const response = await api.get('/api/common/shelters');
        setShelters(response.data || []);
      } catch (error) {
        console.error('보호소 정보 로딩 실패:', error);
        setShelters([]);
      } finally {
        setSheltersLoading(false);
      }
    };

    if (isOpen) {
      loadShelters();
    }
  }, [isOpen]);

  if (!isOpen || !pet) return null;

  // 품종 코드를 품종명으로 변환하는 함수
  const getBreedName = (breedCode) => {
    if (!breedCode) return '-';
    
    if (codesLoading) {
      return `품종코드 ${breedCode}`;
    }
    
    if (codesError || !codes || codes.length === 0) {
      return `품종코드 ${breedCode}`;
    }
    
    // DOG_BREED 그룹에서 해당 코드 찾기
    const breedInfo = codes.find(code => 
      code.groupCd === 'DOG_BREED' && 
      code.cd === breedCode && 
      code.useYn === 'Y'
    );
    
    return breedInfo ? breedInfo.cdNm : `품종코드 ${breedCode}`;
  };

  // 보호소 ID를 보호소명으로 변환하는 함수
  const getShelterName = (shelterId) => {
    if (!shelterId) return '-';
    
    if (sheltersLoading) {
      return `보호소ID ${shelterId}`;
    }
    
    if (!shelters || shelters.length === 0) {
      return `보호소ID ${shelterId}`;
    }
    
    // 보호소 목록에서 해당 ID 찾기
    const shelterInfo = shelters.find(shelter => 
      String(shelter.shelterId) === String(shelterId)
    );
    
    return shelterInfo ? shelterInfo.shelterName : `보호소ID ${shelterId}`;
  };

  // 성별 변환
  const getGenderText = (genderCode) => {
    switch (genderCode) {
      case 'F': return '암컷';
      case 'M': return '수컷';
      case 'Q': return '미상';
      default: return '-';
    }
  };

  // 중성화 여부 변환
  const getNeuterText = (neuterCode) => {
    switch (neuterCode) {
      case 'Y': return '완료';
      case 'N': return '미완료';
      case 'U': return '미상';
      default: return '-';
    }
  };

  // 상태 변환
  const getStatusText = (statusCode) => {
    switch (statusCode) {
      case 'APPLY_AVAILABLE': return '입양 신청 가능';
      case 'ADOPTED': return '입양완료';
      case 'DEAD': return '자연사';
      case 'RETURN': return '기증자반환';
      default: return statusCode || '-';
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* 헤더 */}
        <div className="sticky top-0 bg-white border-b border-gray-200 p-4 flex justify-between items-center rounded-t-lg">
          <h2 className="text-xl font-bold text-gray-800">🐕 보호동물 상세정보</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 text-2xl font-bold"
          >
            ×
          </button>
        </div>

        {/* 내용 */}
        <div className="p-6">
          {/* 이미지 섹션 */}
          <div className="text-center mb-6">
            <img
              src={pet.public_url}
              alt="보호동물 사진"
              className="w-full max-w-md mx-auto rounded-lg shadow-md"
              onError={(e) => {
                e.target.src = '/dog_care_favicon.png'; // 기본 이미지로 대체
              }}
            />
          </div>

          {/* 기본 정보 */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div className="bg-green-50 p-3 rounded-lg">
              <h3 className="font-semibold text-green-800 mb-2">기본 정보</h3>
              <div className="space-y-1.5 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">품종:</span>
                  <span className="font-medium">{getBreedName(pet.breed_cd)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">성별:</span>
                  <span className="font-medium">{getGenderText(pet.gender_cd)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">나이:</span>
                  <span className="font-medium">{pet.birth_yyyy_mm || '-'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">체중:</span>
                  <span className="font-medium">{pet.weight_kg ? `${pet.weight_kg}kg` : '-'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">털색:</span>
                  <span className="font-medium">{pet.color || '-'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">중성화:</span>
                  <span className="font-medium">{getNeuterText(pet.neutered_cd)}</span>
                </div>
              </div>
            </div>

            <div className="bg-blue-50 p-3 rounded-lg">
              <h3 className="font-semibold text-blue-800 mb-2">보호 정보</h3>
              <div className="space-y-1.5 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">상태:</span>
                  <span className="font-medium">{getStatusText(pet.adoption_status_cd)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">공고번호:</span>
                  <span className="font-medium">{pet.notice_id || '-'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">접수일:</span>
                  <span className="font-medium">{pet.reception_date || '-'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">공고시작일:</span>
                  <span className="font-medium">{pet.notice_start_date || '-'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">공고종료일:</span>
                  <span className="font-medium">{pet.notice_end_date || '-'}</span>
                </div>
              </div>
            </div>
          </div>

          {/* 특징 및 설명 */}
          {pet.feature && (
            <div className="bg-yellow-50 p-4 rounded-lg mb-6">
              <h3 className="font-semibold text-yellow-800 mb-3">🌟 특징</h3>
              <p className="text-sm text-gray-700 whitespace-pre-wrap">{pet.feature}</p>
            </div>
          )}

          {/* 연락처 정보 */}
          <div className="bg-purple-50 p-4 rounded-lg mb-6">
            <h3 className="font-semibold text-purple-800 mb-3">📞 연락처</h3>
            <div className="text-sm text-gray-600 text-center">
              <p className="mb-2">입양 문의는 해당 보호소에 직접 연락해주세요.</p>
              <p className="text-xs">보호소 정보는 별도로 조회하실 수 있습니다.</p>
            </div>
          </div>

          {/* 위치 정보 */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-semibold text-gray-800 mb-3">📍 위치 정보</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">발견장소:</span>
                <span className="font-medium">{pet.found_location || '-'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">보호소:</span>
                <span className="font-medium">{getShelterName(pet.shelter_id)}</span>
              </div>
            </div>
          </div>
        </div>

        {/* 푸터 */}
        <div className="sticky bottom-0 bg-white border-t border-gray-200 p-4 rounded-b-lg">
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="flex-1 bg-gray-300 text-gray-700 py-2 px-4 rounded-lg hover:bg-gray-400 transition-colors font-medium"
            >
              닫기
            </button>
            <button
              onClick={() => {
                alert('보호소 연락처는 별도 조회를 통해 확인해주세요.');
              }}
              className="flex-1 bg-green-500 text-white py-2 px-4 rounded-lg hover:bg-green-600 transition-colors font-medium"
            >
              📞 보호소 문의
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
