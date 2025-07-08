import { useState, useEffect, useMemo } from 'react';
import {
  Dialog,
  DialogHeader,
  DialogBody,
  DialogFooter,
  Button,
  Select,
  Option,
  Input,
  Typography, Textarea,
  ButtonGroup,
} from "@material-tailwind/react";
import api from '@/api/api';
import { useCommonCodes } from '@/contexts/CommonCodeContext';
import AlertDialog from '@/components/common/AlertDialog';

const initFormData = {
    petUid: '',
    name: '',
    birthYyyyMm: '',
    receptionDate: new Date().toISOString().slice(0, 10),
    weightKg: '',
    color: '',
    genderCd: 'Q',
    breedCd: '307',
    neuteredCd: 'U',
    adoptionStatusCd: 'PREPARING',
    jurisdictionOrg: 'ALL',
    shelterId: '',
    foundLocation: '',
    feature: '',
    noticeId: '',
}

export default function PetRegister({ isOpen, onClose, onSaved, petUid}) {
  const { codes, loading: codesLoading, error } = useCommonCodes(); // 전체 공통코드 목록
  const [shelters, setShelters] = useState([]);
  const [progressConfig, setProgressConfig] = useState({isLoading: false, message: ''});
  const [isEditMode, setIsEditMode] = useState(false);  // 수정 모드 여부
  const [formData, setFormData] = useState(initFormData);
  const [files, setFiles] = useState([]);
  const [deletedFileIds, setDeletedFileIds] = useState([]);
  const [alertOption, setAlertOption] = useState({isOpen: false, message: '', title: ''});
  const [isUpdated, setIsUpdated] = useState(false);
  
  useEffect(() => {
    console.log('>>>> petUid', petUid);
    if (isOpen) {
      initializeData();
    }
  }, [isOpen, petUid]);

  const initializeData = async () => {
    setIsEditMode(() => petUid);  // petUid가 있으면 수정 모드
    setProgressConfig({isLoading: true, message: '데이터 로딩 중...'});
    
    try {
      // 1. 보호소 목록 조회
      const shelterResponse = await api.get('/api/common/shelters');
      setShelters(shelterResponse.data.map(o => ({
        ...o,
        shelterId: String(o.shelterId),
      })));
      
      // 2. petUid가 있으면 기존 데이터 조회, 없으면 초기화
      if (petUid) {
        await loadPetData(petUid);
      } else {
        resetFormData();
      }
      
    } catch (error) {
      console.error('초기 데이터 로딩 실패:', error);
      setAlertOption({
        isOpen: true, 
        message: '데이터 로딩 중 오류가 발생했습니다.', 
        title: '로딩실패'
      });
    } finally {
      setProgressConfig({isLoading: false, message: ''});
    }
  };

  const loadPetData = async (petUid) => {
    try {
      const response = await api.get(`/manager/pet/${petUid}`);
      const petData = response?.data?.petProfile || {};
      const petImages = response?.data?.petImages || [];
      
      setFormData({
        ...initFormData,
        ...petData,
        receptionDate: petData.receptionDate || new Date().toISOString().slice(0, 10),
      });

      if (Array.isArray(petImages) && petImages.length > 0) {
        setFiles(petImages.map(img => ({
          petImageId: img.petImageId,
          url: img.imageUrl,
          name: img.fileName,
          type: img.fileType,
          size: img.fileSize,
          isBestImage: img?.isBestImage === 'Y' || false, // 대표이미지 여부
        })));
      } else {
        setFiles([]);
      }
      
    } catch (error) {
      console.error('펫 데이터 조회 실패:', error);
      setAlertOption({
        isOpen: true,
        message: '펫 정보를 불러오는데 실패했습니다.',
        title: '조회 실패'
      });
    }
  };

  const resetFormData = () => {
    setFormData(initFormData);
    setFiles([]);
  };

  // shelterId 자동설정
  useEffect(() => {
    if (shelters.length > 0 && !isEditMode && !formData.shelterId) {
      const first = shelters[0];
      setFormData(f => ({
        ...f,
        shelterId: first.shelterId,
      }));
    }
  }, [shelters, isEditMode]);

  // 관할지역 변경 시 해당 보호소 자동 선택
  useEffect(() => {
    if (shelters.length > 0 && formData.jurisdictionOrg) {
      const currentFilteredShelters = (formData.jurisdictionOrg === 'ALL') 
        ? shelters 
        : shelters.filter(s => String(s.jurisdictionOrg) === formData.jurisdictionOrg);
      
      // 현재 선택된 보호소가 필터링된 목록에 없으면 첫 번째 보호소로 변경
      if (currentFilteredShelters.length > 0) {
        const currentShelterExists = currentFilteredShelters.some(s => s.shelterId === formData.shelterId);
        if (!currentShelterExists) {
          setFormData(prev => ({
            ...prev,
            shelterId: currentFilteredShelters[0].shelterId,
            foundLocation: currentFilteredShelters[0].shelterRoadAddr,
          }));
        }
      }
    }
  }, [formData.jurisdictionOrg, shelters]);

  // 현재 선택된 관할지역에 따른 보호소 목록 필터링
  const filteredShelters = useMemo(() => {
    return (formData.jurisdictionOrg === 'ALL' || formData.jurisdictionOrg === '전체') 
      ? shelters 
      : shelters.filter(s => String(s.jurisdictionOrg) === formData.jurisdictionOrg);
  }, [formData.jurisdictionOrg, shelters]);

  const OPTIONS = {
    GENDER: codes?.filter(code => code.groupCd === 'PET_GENDER'),
    NEUTER: codes?.filter(code => code.groupCd === 'NEUTER_STATUS'),
    ADOPTION_STATUS: codes?.filter(code => code.groupCd === 'ADOPTION_STATUS'),
    BREED: codes?.filter(code => code.groupCd === 'DOG_BREED'),
    // COLOR: codes?.filter(code => code.groupCd === 'PET_COLOR'),
    SHELTER: shelters,
    ORG: ['전체', ...new Set(shelters.map(o => String(o.jurisdictionOrg)))],
  };
  
  // if (isLoading) return null; // 모든 훅 호출 이후에 로딩 상태 체크

  const handleJurisdictionChange = (selectedOrg) => {
    setFormData(prev => ({
      ...prev,
      jurisdictionOrg: selectedOrg
    }));
  };

  const handleChange = (e) => {
    const { name, value, files } = e.target;
    const maxFiles = 5;
    if (name === 'image') {
      const fileArray = Array.from(files);
      setFiles(prev => {
        const remainingSlots = maxFiles - prev.length;
        if (remainingSlots <= 0) return prev;
        return [...prev, ...fileArray.slice(0, remainingSlots)];
      });
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };

  const validateForm = () => {    
    const errs = [];
    if (files.length < 1) errs.push('사진을 최소 1장 이상 업로드해주세요.');
    if (files.length > 5) errs.push('사진은 최대 5장까지만 업로드 가능합니다.');
    //if (formData.name === '') errs.push('이름을 입력해주세요.');
    //if (formData.birthYyyyMm === '') errs.push('출생년월을 입력해주세요.');
    if (formData.breedCd === '') errs.push('견종을 선택해주세요.');
    if (formData.genderCd === '') errs.push('성별을 선택해주세요.');
    if (formData.color === '') errs.push('색상을 입력해주세요.');
    //if (formData.weightKg === '') errs.push('체중을 입력해주세요.');
    if (formData.neuteredCd === '') errs.push('중성화 여부를 선택해주세요.');
    if (formData.adoptionStatusCd === '') errs.push('입양 상태를 선택해주세요.');
    if (['', 'ALL'].includes(formData.jurisdictionOrg)) errs.push('관할지역을 선택해주세요.');
    if (formData.shelterId === '') errs.push('보호센터를 선택해주세요.');
    if (formData.receptionDate === '') errs.push('접수일자를 입력해주세요.');
    return errs;
  };

  const handleGenProfile = async (e) => {
    if (e) e.preventDefault();
    if (!_validateForm()) return;

    try {
      setProgressConfig({isLoading: true, message: '프로필 페이지 생성 중...'});
      const res = await api.get(`/manager/gen-profile/${formData.petUid}`)
      .then(res => {
        // _openNewTab(res?.data);
        if (!res?.data?.success) {
          setAlertOption({isOpen: true, message: res?.data?.error || '프로필 생성 실패', title: '프로필 생성 실패'});
        } else {
          setFormData(f => ({...f, profileHtml: res?.data?.htmlContent || ''}));
          setIsUpdated(true);
          setAlertOption({isOpen: true, message: '프로필 컨텐츠가 생성되었습니다.', title: '프로필 생성 완료'});
        }
      })
      .catch(e => {
        console.error(e);
        throw e;
      });
    } catch (err) {
      console.error(err);
      setAlertOption({isOpen: true, message: '프로필 생성 중 오류가 발생했습니다.', title: '프로필 생성 오류'});
    } finally {
      setProgressConfig({isLoading: false, message: ''});
    }
  }

  const _openNewTab = (html) => {
    try {
      // Blob을 사용해서 URL 생성
      const blob = new Blob([html], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      
      // 새 탭에서 열기
      const newWindow = window.open(url, '_blank');
      
      if (newWindow) {
        // 메모리 정리 (5초 후)
        setTimeout(() => URL.revokeObjectURL(url), 5000);
      } else {
        setAlertOption({isOpen: true, message: '팝업을 허용해야 보실 수 있습니다.', title: '알림'});
        // 팝업 차단 시 다운로드로 대체
        // const link = document.createElement('a');
        // link.href = url;
        // link.download = 'result.html';
        // link.click();
        // URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('HTML 표시 실패:', error);
    }
  }

  const _validateForm = () => {
    const errs = validateForm();
    if (errs.length > 0) {
      const errHtml = `
      <ul className="text-sm text-amber-600 list-disc list-inside">
        ${errs.map((e, i) => `<li key=${i}>${e}</li>`).join('')}
      </ul>`
      setAlertOption({isOpen: true, message: errHtml, title: '입력 체크'});
      return false;
    }
    return true
  }

  const handleImageClick = (file) => {
    const imageUrl = file.url || URL.createObjectURL(file);
    window.open(imageUrl, '_blank', 'width=800,height=600,scrollbars=yes,resizable=yes');
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!_validateForm()) return;
    
    try {
      setProgressConfig({isLoading: true, message: `서버 ${isEditMode ? '저장' : '등록'} 중... `});

      const formDataToSend = new FormData();
      formDataToSend.append('petProfile', new Blob([JSON.stringify(formData)], { type: 'application/json' }));
      
      // File 객체인 경우만 FormData에 추가 (기존 이미지는 제외)
      files.forEach(file => {
        if (file instanceof File) {
          formDataToSend.append('photos', file);
          // "name": "00156db4a-1.jpg",
          // "size": 13293,
          // "type": "image/jpeg"
        } else {
          // petImageId: img.petImageId,
          // url: img.imageUrl,
          // name: img.fileName,
          // type: img.fileType,
          // size: img.fileSize,
        }
      });
      formDataToSend.append('deleteImageIds', new Blob([JSON.stringify(deletedFileIds)], { type: 'application/json' }));

      
      // 수정 모드면 PUT, 신규면 POST
      const response = isEditMode 
        ? await api.put(`/manager/pet`, formDataToSend)
        : await api.post('/manager/pet', formDataToSend);
        
      onSaved();
      
    } catch (err) {
      console.error(err);
      const message = isEditMode ? '수정 중 오류 발생' : '등록 중 오류 발생';
      setAlertOption({
        isOpen: true, 
        message: err.response?.data?.error || message, 
        title: isEditMode ? '수정 오류' : '등록 오류'
      });
    } finally {
      setProgressConfig({isLoading: false, message: ''});
    }
  };

  // TODO: 디버깅용
  const _consoleFormData = (data) => {
    const result = {};
    
    for (let [key, value] of data.entries()) {
      if (key === 'photos') {
        // photos 배열이 없으면 생성
        if (!result.photos) {
          result.photos = [];
        }
        result.photos.push({
          name: value.name,
          size: value.size,
          sizeKB: `${(value.size / 1024).toFixed(2)} KB`,
          type: value.type,
          lastModified: new Date(value.lastModified).toLocaleString()
        });
      } else {
        result[key] = value;
      }
    }    
    // console.log("formData => ", result);
    //console.log("formData => ", JSON.stringify(result, null, 2));
  }

  const handleFileDelete = (idx) => {
    const fileToDelete = files[idx];
    
    // 기존 이미지인 경우 deletedFileIds에 추가
    if (fileToDelete.petImageId) {
      setDeletedFileIds(prev => [...prev, fileToDelete.petImageId]);
    }
    
    // files에서 해당 파일 제거
    setFiles(prev => prev.filter((_, i) => i !== idx));
  };

  return (
    <Dialog open={isOpen} handler={onClose} size="lg" className="z-50 h-[90vh] flex flex-col">
      <DialogHeader className="flex justify-between items-center">
        <span className="text-xl font-semibold">
          {isEditMode ? '보호견 수정' : '보호견 등록'}
        </span>
        <Button variant="text" color="blue" size="sm" onClick={onClose} className="p-1">✕</Button>
      </DialogHeader>

      <DialogBody className="flex-1 overflow-y-auto px-4 scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-gray-100">
        <div className="relative">
          {/* 로딩 오버레이 */}
          {progressConfig.isLoading && (
            <div className="absolute inset-0 bg-white bg-opacity-80 flex flex-col items-center justify-center z-10 rounded-md">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4"></div>
              <Typography variant="h6" color="blue-gray">
                {progressConfig.message}
              </Typography>
            </div>
          )}
          
          {/* 메인 폼 - 항상 렌더링 */}
          <form id="pet-register-form" className="space-y-4 pr-2">
            {/* 이미지 업로드 및 미리보기 */}
            <div className="space-y-2">
              <div className="grid" style={{ gridTemplateColumns: '120px 1fr', alignItems: 'center', gap: '8px' }}>
                <label className="font-medium text-right">사진 업로드</label>
                <div className="space-y-1">
                <input
                  type="file"
                  name="image"
                  multiple
                  accept="image/*"
                  onChange={handleChange}
                  className="block w-full text-sm text-gray-600 file:py-1 file:px-2 file:border-0 file:rounded file:bg-blue-50 file:text-blue-700"
                />
                  <Typography variant="small" className="text-gray-500">
                    최대 5장까지 업로드 가능 ({files.length}/5)
                  </Typography>
                </div>
              </div>

              {/* 미리보기 */}
              <div className="w-full h-[100px] border-2 border-dashed border-gray-300 rounded-md overflow-x-auto flex items-center gap-2 p-2">
                {files.length > 0 ? (
                  files.map((file, idx) => (
                    <div key={idx} className="relative flex-shrink-0">
                    <img
                        src={file.url || URL.createObjectURL(file)}
                      alt={`preview-${idx}`}
                        className="h-full w-auto object-contain rounded cursor-pointer"
                        style={{ height: '80px' }}
                        onClick={() => handleImageClick(file)}
                      />
                      {/* 대표이미지 뱃지 */}
                      {file.isBestImage && (
                        <div className="absolute top-0 left-0 bg-yellow-400 text-yellow-900 text-xs px-1 rounded-br-md rounded-tl-md font-semibold">
                          👑 대표
                        </div>
                      )}
                      <button
                        type="button"
                        onClick={() => handleFileDelete(idx)}
                        className="absolute -top-1 -right-1 bg-red-500 text-white rounded-full w-4 h-4 text-xs flex items-center justify-center"
                      >
                        ×
                      </button>
                    </div>
                  ))
                ) : (
                  <span className="text-sm text-gray-400 w-full text-center">미리보기</span>
                )}
              </div>
            </div>

            {/* 입력 필드 */}
            <div className="grid grid-cols-2 gap-3">
              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">이름</Typography>
                <Input
                  name="name"
                  value={formData.name || ''}
                  onChange={handleChange}
                  className="w-full"
                />
              </div>
              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">출생년월</Typography>
                <Input
                  label="출생년월(2025-01)"
                  name="birthYyyyMm"
                  type="month"
                  value={formData.birthYyyyMm || ''}
                  onChange={handleChange}
                  className="w-full"
                />
              </div>
              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">
                  <span className="text-red-500">*</span>견종
                </Typography>
                <Select 
                  name="breedCd" 
                  value={formData.breedCd} 
                  onChange={(val) => setFormData(f => ({ ...f, breedCd: val }))} 
                  required
                  filter
                  filterOption={(input, option) => 
                    option.children.toLowerCase().includes(input.toLowerCase())
                  }
                >
                  {OPTIONS.BREED
                    ?.sort((a, b) => a.cdNm.localeCompare(b.cdNm))
                    .map(o => <Option key={o.cd} value={o.cd}>{o.cdNm}</Option>)
                  }
                </Select>
              </div>
              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">
                  <span className="text-red-500">*</span>성별
                </Typography>
                <Select name="genderCd" value={formData.genderCd} onChange={(val) => setFormData(f => ({ ...f, genderCd: val }))} required>
                  {OPTIONS.GENDER.map(o => <Option key={o.cd} value={o.cd}>{o.cdNm}</Option>)}
                </Select>
              </div>
              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">
                  <span className="text-red-500">*</span>색상
                </Typography>
                {/* <Select name="color" value={formData.color} onChange={(val) => setFormData(f => ({ ...f, color: val }))} required>
                  {OPTIONS.COLOR.map(o => <Option key={o.cdNm} value={o.cdNm}>{o.cdNm}</Option>)}
                </Select> */}
                <Input
                  label="갈색&흰색"
                  name="color"
                  type="text"
                  value={formData.color || ''}
                  onChange={handleChange}
                  className="w-full"
                />
              </div>
              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">체중(kg)</Typography>
                <Input
                  label="숫자만(입력예: 4.8)"
                  name="weightKg"
                  type="number"
                  step="0.1"
                  value={formData.weightKg || ''}
                  onChange={handleChange}
                  className="w-full"
                />
              </div>
              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">
                  <span className="text-red-500">*</span>중성화
                </Typography>
                <Select name="neuteredCd" value={formData.neuteredCd} onChange={(val) => setFormData(f => ({ ...f, neuteredCd: val }))} required>
                  {OPTIONS.NEUTER.map(o => <Option key={o.cd} value={o.cd}>{o.cdNm}</Option>)}
                </Select>
              </div>
              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">
                  <span className="text-red-500">*</span>입양 상태
                </Typography>
                <Select name="adoptionStatusCd" value={formData.adoptionStatusCd} 
                    onChange={(val) => setFormData(f => ({ ...f, adoptionStatusCd: val }))} required>
                  {OPTIONS.ADOPTION_STATUS.map(o => <Option key={o.cd} value={o.cd}>{o.cdNm}</Option>)}
                </Select>
              </div>
              
              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">
                  <span className="text-red-500">*</span>관할지역
                </Typography>
                <Select 
                  key={`jurisdiction-${OPTIONS.ORG.length}`}
                  name="jurisdictionOrg" 
                  value={formData.jurisdictionOrg === 'ALL' ? '전체' : formData.jurisdictionOrg} 
                  onChange={(val) => handleJurisdictionChange(val === '전체' ? 'ALL' : val)}>
                  {OPTIONS.ORG.map(org => (
                    <Option key={org} value={org}>
                      {org}
                    </Option>
                  ))}
                </Select>
              </div>
              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">
                  <span className="text-red-500">*</span>보호센터
                </Typography>
                <Select 
                  key={`shelter-${formData.jurisdictionOrg}-${filteredShelters.length}`}
                  name="shelterId" 
                  value={formData.shelterId} 
                  onChange={(val) => setFormData(f => ({ ...f, shelterId: val, foundLocation: filteredShelters.find(s => s.shelterId === val)?.shelterRoadAddr }))} 
                  required>
                  {filteredShelters.map(shelter => (
                    <Option key={shelter.shelterId} value={String(shelter.shelterId)}>
                      {shelter.shelterName}
                    </Option>
                  ))}
                </Select>
              </div>
            </div>

            <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
              <Typography variant="small" className="text-right font-medium text-blue-gray-900">
                발견 장소
              </Typography>
              <Input
                label="발견 주소 및 장소(예: 서울시 강남구 역삼동 123-45 현대백화점 앞)"
                name="foundLocation"
                type="text"
                value={formData.foundLocation || ''}
                onChange={handleChange}
                className="w-full"
                maxLength={255}
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">
                  <span className="text-red-500">*</span>접수일자
                </Typography>
                <Input
                  name="receptionDate"
                  type="date"
                  value={formData.receptionDate}
                  onChange={handleChange}
                  className="w-full"
                  readOnly
                />
              </div>
              {isEditMode ? (
              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                  <><Typography variant="small" className="text-right font-medium text-blue-gray-900">
                    <span className="text-red-500">*</span>펫ID
                  </Typography>
                  <Input
                      name="noticeId"
                      type="text"
                      value={formData.noticeId || ''}
                      className="w-full bg-gray-100 text-gray-600"
                      readOnly
                      label=""
                      hidden={isEditMode ? false : true} /></>
                </div>
              ) : ``}
            </div>

            {/* 특징 필드 - 전체 너비 */}
            <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
              <Typography variant="small" className="text-right font-medium text-blue-gray-900">
                특징
              </Typography>
              <Input
                label="최대 1000자(예: 70일 추정, 귀엽고 사랑스러움)"
                name="feature"
                type="text"
                value={formData.feature || ''}
                onChange={handleChange}
                className="w-full"
                maxLength={1000}
                rows={3}
                style={{height: "60px"}}
              />
            </div>
          </form>
        </div>
      </DialogBody>
      <DialogFooter className="flex justify-end gap-2 px-4 flex-shrink-0 border-t border-gray-200 bg-gray-50">
        {/* <Button onClick={() => {console.log(files, deletedFileIds); console.log(formData)}}>파일 확인</Button> */}
        <Button variant="text" className="bg-gray-500 hover:bg-gray-400 text-white" 
          onClick={() => isUpdated ? onSaved() : onClose()}>
          닫기
        </Button>
        {isEditMode ? (
        <ButtonGroup variant="filled" size="md" color="amber">
          <Button 
            onClick={() => _openNewTab(formData.profileHtml)}             
            disabled={progressConfig.isLoading || formData.profileHtml === null}
          >
            프로필 열람
          </Button>
          <Button 
            onClick={handleGenProfile} 
            disabled={progressConfig.isLoading}
          >
            프로필 생성
          </Button>
        </ButtonGroup>
        ) : ``}
        <Button 
          onClick={handleSubmit} 
          variant="text" 
          className="bg-blue-700 hover:bg-blue-300 text-white"
          disabled={progressConfig.isLoading}
        >
          {isEditMode ? '저장' : '등록'}
        </Button>
      </DialogFooter>
      <AlertDialog 
        isOpen={alertOption.isOpen} 
        onClose={() => setAlertOption({isOpen: false, message: '', title: ''})} 
        message={alertOption.message} 
        title={alertOption.title} 
      />
    </Dialog>
  );
}