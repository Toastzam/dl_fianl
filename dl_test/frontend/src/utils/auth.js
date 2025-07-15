// JWT 토큰 유효성 검증 유틸리티

/**
 * JWT 토큰의 만료 시간을 확인하는 함수
 * @param {string} token - JWT 토큰
 * @returns {boolean} - 토큰이 유효하면 true, 만료되었거나 잘못된 토큰이면 false
 */



import { API_SERVER_URL } from '@/config/env';

export const isTokenValid = (token) => {
  if (!token) return false;
  
  try {
    // JWT 토큰을 디코딩 (base64)
    const payload = JSON.parse(atob(token.split('.')[1]));
    const currentTime = Math.floor(Date.now() / 1000); // 현재 시간 (초 단위)
    
    // exp (expiration time) 필드가 있는지 확인하고 만료 여부 검사
    if (payload.exp && payload.exp < currentTime) {
      console.log('[Token Validation] 토큰이 만료되었습니다.');
      return false;
    }
    
    return true;
  } catch (error) {
    console.log('[Token Validation] 토큰 형식이 잘못되었습니다:', error);
    return false;
  }
};

/**
 * 백엔드 API를 통해 토큰 유효성을 검증하는 함수
 * @param {string} token - 검증할 JWT 토큰
 * @returns {Promise<boolean>} - 토큰이 유효하면 true
 */
export const validateTokenWithServer = async (token) => {
  if (!token) return false;
  
  try {
    const response = await fetch(`${API_SERVER_URL}/api/user/me`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });
    
    // 200 응답이면 토큰이 유효함
    return response.ok;
  } catch (error) {
    console.log('[Server Token Validation] 토큰 검증 실패:', error);
    return false;
  }
};

/**
 * localStorage에서 토큰을 가져와 유효성을 검증하는 함수
 * @param {boolean} useServerValidation - 서버 검증 사용 여부 (기본값: false)
 * @returns {Promise<boolean>} - 토큰이 유효하면 true
 */
export const checkAuthStatus = async (useServerValidation = false) => {
  const token = localStorage.getItem('token');
  const isLoggedIn = localStorage.getItem('isLoggedIn') === 'true';
  
  if (!token || !isLoggedIn) {
    return false;
  }
  
  // 클라이언트 사이드 검증
  if (!isTokenValid(token)) {
    // 만료된 토큰이면 localStorage 정리
    clearAuthData();
    return false;
  }
  
  // 서버 사이드 검증 (옵션)
  if (useServerValidation) {
    const isServerValid = await validateTokenWithServer(token);
    if (!isServerValid) {
      clearAuthData();
      return false;
    }
  }
  
  return true;
};

/**
 * 인증 관련 localStorage 데이터를 모두 정리하는 함수
 */
export const clearAuthData = () => {
  localStorage.removeItem('token');
  localStorage.removeItem('isLoggedIn');
  localStorage.removeItem('userId');
  localStorage.removeItem('name');
  localStorage.removeItem('role');
  console.log('[Auth] 인증 데이터가 정리되었습니다.');
};

/**
 * 사용자의 역할을 안전하게 가져오는 함수
 * @returns {Promise<string>} - 사용자 역할 ('USER', 'ADMIN', 또는 '')
 */
export const getUserRole = async () => {
  const isAuthenticated = await checkAuthStatus();
  if (!isAuthenticated) {
    return '';
  }
  
  return localStorage.getItem('role') || '';
};
