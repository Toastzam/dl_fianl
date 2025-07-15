/**
 * Axios 인스턴스 분리(Request & Response Interceptor)
 */
import axios from 'axios';
import { API_SERVER_URL, NODE_ENV } from '@/config/env';

const api = axios.create({
  baseURL: API_SERVER_URL,
  timeout: 20000,
  // headers: {'Content-Type': 'application/json'} // 기본 헤더
});

// 요청 인터셉터 (Request Interceptor): 모든 요청이 보내지기 전 실행
api.interceptors.request.use(
  (config) => {
    // JWT 토큰을 헤더에 자동 추가
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    if (NODE_ENV === 'local') {
      console.debug(`[API Request] ${config.method.toUpperCase()} ${config.url}`, config);
    }
    return config;
  },
  (error) => {
    console.error('[API Request] error:', error);
    return Promise.reject(error);
  }
);

// 응답 인터셉터 (Response Interceptor): 모든 응답을 받은 후에 실행
api.interceptors.response.use(
  (response) => {
    if (NODE_ENV === 'local') {
      console.debug(`[API] Response:`, response);
    }
    return response;
  },
  (error) => {
    const { response, config } = error;
    
    if (NODE_ENV === 'local') {
      console.error(`[API Response] Error:`, error);
      console.error(`[API Response] Error Response:`, response);
      // console.error(`[API] Error Response Data:`, JSON.stringify(response?.data, null, 2));
      // console.error(`[API] Error Config:`, config);
    }
    
    if (!response) {
      // 네트워크 에러, 타임아웃 등
      console.error('네트워크 오류가 발생했습니다. 다시 시도해주세요.');
      alert('서버에 연결할 수 없습니다. 네트워크 연결을 확인해주세요.');
      return Promise.reject(error);
    }

    switch (response.status) {
      case 400:
        console.warn('잘못된 요청입니다.');
        break;

      case 401:
        // 로그인 요청이 아닌 경우에만 토큰 만료 처리
        if (!config.url.includes('/login')) {
          console.warn('인증에 실패했습니다. 다시 로그인해 주세요.');
          localStorage.removeItem('token');
          localStorage.removeItem('isLoggedIn');
          localStorage.removeItem('userId');
          localStorage.removeItem('name');
          localStorage.removeItem('role'); // role 추가
          alert('로그인이 만료되었습니다. 다시 로그인해 주세요.');
          window.location.href = '/';
        }
        break;

      case 403:
        console.error('접근 권한이 없습니다.');
        break;

      case 404:
        console.info('요청하신 리소스를 찾을 수 없습니다.');
        break;

      case 500:
      default:
        console.error('서버에 오류가 발생했습니다.');
        break;
    }    
    // 2xx 범위를 벗어나는 상태 코드에 대한 에러 처리
    // 예: 401 Unauthorized 에러 시 로그인 페이지로 리다이렉트
    // if (error.response && error.response.status === 401) {
    //   // 로그아웃 처리 또는 로그인 페이지로 이동
    // }
    return Promise.reject(error);
  }
);

export default api; 

// API 함수들
export const checkUserIdDuplicate = (userId) => {
  return api.get(`/api/user/check-duplicate/${userId}`);
};

export const checkEmailDuplicate = (email) => {
  return api.get(`/api/user/check-duplicate-email/${email}`);
};

export const registerUser = (userData) => {
  return api.post('/api/user/register', userData);
};

export const loginUser = (userData) => {
  return api.post('/api/user/login', userData);
};

// 비밀번호 변경 API
export function changePassword({ currentPassword, newPassword }) {
  return api.patch('/api/user/password', { currentPassword, newPassword });
}

// 내 정보 조회 API
export function getMyInfo() {
  return api.get('/api/user/me');
}

// 전화번호 변경 API
export function updatePhone({ phone }) {
  return api.patch('/api/user/phone', { phone });
}