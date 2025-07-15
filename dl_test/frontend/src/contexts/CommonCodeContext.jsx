// 공통코드 Context + Provider
import { createContext, useContext, useEffect, useState } from 'react';
import api from '../api/api';

const CommonCodeContext = createContext(null);

export function CommonCodeProvider({ children }) {
  const [codes, setCodes] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadCodes = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get('/api/common/codes?useYn=Y');
      setCodes(res.data);
    } catch (err) {
      setError(err);
      console.error('공통코드 로딩 실패:', err);
    } finally {
      setLoading(false);
    }
  };

  const reloadCodes = async () => {
    await loadCodes();
  };

  useEffect(() => {
    loadCodes();
  }, []);

  return (
    <CommonCodeContext.Provider value={{ codes, loading, error, reloadCodes }}>
      {children}
    </CommonCodeContext.Provider>
  );
}

export function useCommonCodes() {
  return useContext(CommonCodeContext);
}

// 편의 함수들을 포함한 훅
export function useCommonCode() {
  const { codes, loading, error, reloadCodes } = useContext(CommonCodeContext);
  
  // 품종 코드에서 이름 가져오기
  const getBreedName = (breedCode) => {
    if (!codes || !breedCode) return breedCode || '-';
    
    const breed = codes.find(code => code.cd === breedCode && code.groupCd === 'DOG_BREED');
    return breed ? breed.cdNm : breedCode;
  };
  
  // 보호소 코드에서 이름 가져오기
  const getShelterName = (shelterCode) => {
    if (!codes || !shelterCode) return shelterCode || '-';
    const shelter = codes.find(code => code.cd === shelterCode && code.groupCd === 'SHELTER');
    return shelter ? shelter.cdNm : shelterCode;
  };
  
  return {
    codes,
    loading,
    error,
    reloadCodes,
    getBreedName,
    getShelterName
  };
}