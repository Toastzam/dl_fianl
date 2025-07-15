// Vite 환경 변수 래퍼
// 사용법: import해서 상수처럼 사용
// import { API_SERVER_URL, WEBAPP_URL, AI_SEARCH_SERVER_URL } from '@/config/env';

// 실행환경 설정 (local, development, production)
export const NODE_ENV = import.meta.env.NODE_ENV || 'local';

// 서버 주소 (.env 파일에서 가져오기)
export const API_SERVER_URL = import.meta.env.VITE_API_SERVER_URL || "http://localhost:8080";
export const WEBAPP_URL = import.meta.env.VITE_WEBAPP_URL || "http://localhost:5173";

// AI 검색 서버 주소 (별도 서버)
export const AI_SEARCH_SERVER_URL = import.meta.env.VITE_AI_SEARCH_SERVER_URL || "http://localhost:8001";

// 하드코딩된 주소 (백업용, 주석 처리)
// export const API_SERVER_URL = "http://192.168.0.13:8080"; // 백엔드 서버의 실제 IP:포트로 변경
// export const WEBAPP_URL = "http://192.168.0.13:5173"; // 프론트엔드 실제 주소