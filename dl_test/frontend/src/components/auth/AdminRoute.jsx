import React, { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { checkAuthStatus, getUserRole } from '@/utils/auth';

const AdminRoute = ({ children }) => {
  const [authState, setAuthState] = useState({
    isAuthenticated: null, // null: 검증 중, true: 인증됨, false: 인증 실패
    userRole: null // null: 확인 중, 'USER'/'ADMIN': 역할 확인됨
  });

  useEffect(() => {
    const validateAdminAuth = async () => {
      console.log('[AdminRoute] 인증 및 권한 확인 시작');
      
      // 1. 토큰 유효성 검증
      const isValid = await checkAuthStatus(true); // 서버 검증 포함
      if (!isValid) {
        console.log('[AdminRoute] 토큰 검증 실패');
        setAuthState({ isAuthenticated: false, userRole: null });
        return;
      }
      
      // 2. 사용자 역할 확인
      const role = await getUserRole();
      console.log('[AdminRoute] 확인된 사용자 역할:', role);
      
      setAuthState({ 
        isAuthenticated: true, 
        userRole: role 
      });
    };

    validateAdminAuth();
  }, []);

  // 검증 중이면 로딩 표시
  if (authState.isAuthenticated === null || authState.userRole === null) {
    return <div>권한 확인 중...</div>;
  }

  // 로그인하지 않은 경우
  if (!authState.isAuthenticated) {
    alert('로그인이 필요합니다.');
    return <Navigate to="/login" replace />;
  }
  
  // ADMIN 권한이 없는 경우
  if (authState.userRole !== 'ADMIN') {
    alert('관리자 권한이 필요합니다.');
    return <Navigate to="/" replace />;
  }
  
  // ADMIN 권한이 있는 경우 자식 컴포넌트 렌더링
  return children;
};

export default AdminRoute;
