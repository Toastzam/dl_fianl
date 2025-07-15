import React, { useState } from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  CardFooter,
  Typography,
  Button,
  Input,
} from "@material-tailwind/react";
import { loginUser } from '@/api/api';


import { API_SERVER_URL } from '@/config/env';


// 상수 정의
const BASE_URL = API_SERVER_URL || 'http://localhost:8080';
const STORAGE_KEYS = {
  IS_LOGGED_IN: 'isLoggedIn',
  USER_ID: 'userId',
  TOKEN: 'token',
  NAME: 'name',
  ROLE: 'role',
  LOGOUT: 'logout'
};

// 소셜 로그인 데이터
const SOCIAL_LOGIN_PROVIDERS = [
  {
    id: 'google',
    name: 'Google로 로그인',
    hoverColor: 'hover:bg-gray-50',
    icon: (
      <svg className="w-5 h-5" viewBox="0 0 24 24">
        <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
        <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
        <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
        <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
      </svg>
    )
  },
  {
    id: 'kakao',
    name: '카카오로 로그인',
    hoverColor: 'hover:bg-yellow-50',
    icon: (
      <div className="w-5 h-5 bg-yellow-400 rounded flex items-center justify-center">
        <span className="text-black text-xs font-bold">K</span>
      </div>
    )
  },
  {
    id: 'naver',
    name: '네이버로 로그인',
    hoverColor: 'hover:bg-green-50',
    icon: (
      <div className="w-5 h-5 bg-green-500 rounded flex items-center justify-center">
        <span className="text-white text-xs font-bold">N</span>
      </div>
    )
  }
];

const Login = ({ onClose, onSwitchToSignup }) => {


  const [formData, setFormData] = useState({
    userId: '',
    password: '',
  });
  const [loading, setLoading] = useState(false);

  // localStorage에 사용자 정보 저장
  const saveUserToStorage = (userData) => {
    const { userId, token, name, role } = userData;
    
    localStorage.setItem(STORAGE_KEYS.IS_LOGGED_IN, 'true');
    userId && localStorage.setItem(STORAGE_KEYS.USER_ID, userId);
    token && localStorage.setItem(STORAGE_KEYS.TOKEN, token);
    name && localStorage.setItem(STORAGE_KEYS.NAME, name);
    role && localStorage.setItem(STORAGE_KEYS.ROLE, role);
  };

  // 에러 메시지 처리
  const getErrorMessage = (error) => {
    if (error.response?.status === 401) {
      return error.response.data.message || '아이디 또는 비밀번호가 잘못되었습니다.';
    }
    return error.response?.data?.message || error.message || '알 수 없는 오류가 발생했습니다.';
  };

  // 소셜 로그인 핸들러
  const handleSocialLogin = (provider) => {
    window.location.href = `${BASE_URL}/oauth2/authorization/${provider}`;
  };

  // 입력값 변경 핸들러
  const handleInputChange = (field) => (e) => {
    setFormData(prev => ({ ...prev, [field]: e.target.value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await loginUser(formData);
      
      if (response.data.success) {
        const { message, ...userData } = response.data;
        
        // 사용자 정보 저장
        saveUserToStorage(userData);
        
        alert(message || '로그인 성공!');
        onClose();
        window.location.reload();
      } else {
        alert('로그인 실패: ' + (response.data.message || '알 수 없는 오류'));
      }
    } catch (error) {
      console.error('로그인 오류:', error);
      const errorMessage = getErrorMessage(error);
      alert(`로그인 ${error.response?.status === 401 ? '실패' : '중 오류가 발생했습니다'}: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };

  return (

      <Card className="w-96">
        
      {/* 닫기 버튼 */}
      <button
        onClick={onClose}
        className="absolute top-2 right-2 text-gray-500 hover:text-gray-700"
      >
        ✕
      </button>

        <CardHeader variant="gradient" color="blue" className="mb-4 grid h-28 place-items-center">
          <Typography variant="h3" color="white">
            로그인
          </Typography>
        </CardHeader>
        <form onSubmit={handleSubmit}>
          <CardBody className="flex flex-col gap-4">
            <Input
              type="text"
              label="아이디"
              size="lg"
              value={formData.userId}
              onChange={handleInputChange('userId')}
              required
            />
            <Input
              type="password"
              label="비밀번호"
              size="lg"
              value={formData.password}
              onChange={handleInputChange('password')}
              required
            />
          </CardBody>
          <CardFooter className="pt-0">
            <Button 
              variant="gradient" 
              type="submit" 
              fullWidth
              disabled={loading}
            >
              {loading ? '로그인 중...' : '로그인'}
            </Button>
            
            {/* 구분선 */}
            <div className="flex items-center my-4">
              <div className="flex-1 border-t border-gray-300"></div>
              <span className="px-3 text-gray-500 text-sm">또는</span>
              <div className="flex-1 border-t border-gray-300"></div>
            </div>
            
            {/* 소셜 로그인 버튼들 */}
            <div className="space-y-2">
              {SOCIAL_LOGIN_PROVIDERS.map((provider) => (
                <Button
                  key={provider.id}
                  variant="outlined"
                  fullWidth
                  className={`flex items-center justify-center gap-3 text-gray-700 border-gray-300 ${provider.hoverColor}`}
                  onClick={() => handleSocialLogin(provider.id)}
                >
                  {provider.icon}
                  {provider.name}
                </Button>
              ))}
            </div>
            
            <Typography variant="small" className="mt-4 flex justify-center">
              계정이 없으신가요?{" "}
              <span className="ml-1 font-bold text-blue-500 cursor-pointer" onClick={onSwitchToSignup}>
                회원가입
              </span>
            </Typography>
          </CardFooter>
        </form>
      </Card>
  );
};

export default Login;