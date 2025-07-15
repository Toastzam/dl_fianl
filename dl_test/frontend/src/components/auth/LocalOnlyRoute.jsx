import { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { getMyInfo } from '@/api/api';

/**
 * LocalOnlyRoute - 일반 로그인 사용자 전용 라우트 보호 컴포넌트
 * 
 * 기능:
 * - 소셜 로그인(카카오, 구글, 네이버) 사용자의 접근을 차단
 * - 일반 로그인(LOCAL) 사용자만 접근 허용
 * 
 * 사용 목적:
 * - 마이페이지 비밀번호 수정 기능에서 사용
 * - 소셜 로그인 사용자는 비밀번호가 없으므로 비밀번호 수정 불가
 * 
 * 검증 방식:
 * - /api/user/me API 호출하여 사용자 정보 조회
 * - 응답의 provider 필드가 'LOCAL'인지 확인
 * - 'LOCAL'이 아니면 접근 차단하고 메인페이지로 리디렉션
 * 
 * @param {React.ReactNode} children - 보호할 컴포넌트 (예: ChangePassword)
 * @returns {React.ReactElement} 일반 로그인 사용자면 children, 아니면 Navigate
 */
export default function LocalOnlyRoute({ children }) {
  const [allowed, setAllowed] = useState(null); // null: 검증 중, true: 허용, false: 차단

  useEffect(() => {
    // 사용자 정보를 조회하여 provider 확인
    getMyInfo().then(res => {
      // provider가 'local'인 경우만 허용 (대소문자 구분 없음)
      if (res.data.provider?.toLowerCase() === 'local') {
        setAllowed(true);
      } else {
        setAllowed(false);
      }
    }).catch(() => setAllowed(false)); // API 호출 실패 시 차단
  }, []);

  // 검증 중일 때는 아무것도 렌더링하지 않음 (로딩 상태)
  if (allowed === null) return null;
  
  // 일반 로그인 사용자가 아닌 경우 접근 차단
  if (!allowed) {
    alert('일반 로그인 사용자만 접근 가능합니다.');
    return <Navigate to="/" replace />;
  }
  
  // 일반 로그인 사용자인 경우 자식 컴포넌트 렌더링
  return children;
}
