import React, { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { checkAuthStatus } from '@/utils/auth';

/**
 * 보호된 라우트 컴포넌트
 * JWT 토큰의 유효성을 검증하고, 유효하지 않으면 로그인 페이지로 이동
 */
const ProtectedRoute = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(null); // null: 검증 중, true: 인증됨, false: 인증 실패

  useEffect(() => {
    const validateAuth = async () => {
      const isValid = await checkAuthStatus(true); // 서버 검증 포함
      setIsAuthenticated(isValid);
      
      if (!isValid) {
        alert('로그인이 필요한 서비스입니다.');
      }
    };

    validateAuth();
  }, []);

  // 검증 중이면 로딩 표시
  if (isAuthenticated === null) {
    return <div>인증 확인 중...</div>;
  }

  // 인증 실패하면 메인으로 리다이렉트
  if (!isAuthenticated) {
    return <Navigate to="/" replace />;
  }
  
  // 인증 성공하면 자식 컴포넌트 렌더링
  return children;
};

export default ProtectedRoute;
