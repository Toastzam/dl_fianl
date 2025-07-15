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
import { checkUserIdDuplicate, checkEmailDuplicate, registerUser } from '@/api/api';

// 상수 정의
const VALIDATION_MESSAGES = {
  USER_ID_REQUIRED: '아이디를 입력해주세요.',
  EMAIL_REQUIRED: '이메일을 입력해주세요.',
  EMAIL_FORMAT_INVALID: '올바른 이메일 형식을 입력해주세요.',
  USER_ID_DUPLICATE: '이미 사용 중인 아이디입니다.',
  USER_ID_AVAILABLE: '사용 가능한 아이디입니다.',
  EMAIL_DUPLICATE: '이미 사용 중인 이메일입니다.',
  EMAIL_AVAILABLE: '사용 가능한 이메일입니다.',
  EMAIL_API_NOT_IMPLEMENTED: '이메일 중복 확인을 건너뜁니다. (API 미구현)',
  USER_ID_CHECK_REQUIRED: '아이디 중복 확인을 완료해주세요.',
  EMAIL_CHECK_REQUIRED: '이메일 중복 확인을 완료해주세요.',
  PASSWORD_MISMATCH: '비밀번호가 일치하지 않습니다.',
  SIGNUP_SUCCESS: '회원가입이 완료되었습니다!',
  SIGNUP_ERROR: '회원가입 중 오류가 발생했습니다.'
};

const INITIAL_FORM_DATA = {
  userId: '',
  name: '',
  email: '',
  phone: '',
  password: '',
  confirmPassword: '',
};

const INITIAL_CHECK_STATE = {
  checked: false,
  isValid: false,
  message: ''
};

// 중복 확인 컴포넌트 (함수 외부로 분리)
const DuplicateCheckField = ({ 
  type, 
  label, 
  value, 
  onChange, 
  onCheck, 
  checkState, 
  required = false 
}) => (
  <div className="flex flex-col gap-2">
    <div className="flex gap-2">
      <Input
        type={type}
        label={label}
        size="lg"
        value={value}
        onChange={onChange}
        required={required}
        className="flex-1"
      />
      <Button
        type="button"
        variant="outlined"
        size="lg"
        onClick={onCheck}
        className="whitespace-nowrap"
      >
        중복확인
      </Button>
    </div>
    {checkState.checked && (
      <Typography 
        variant="small" 
        className={`${checkState.isValid ? 'text-green-600' : 'text-red-600'}`}
      >
        {checkState.message}
      </Typography>
    )}
  </div>
);

const Signup = ({ onClose, onSwitchToLogin }) => {
  const [formData, setFormData] = useState(INITIAL_FORM_DATA);
  const [loading, setLoading] = useState(false);
  const [userIdCheck, setUserIdCheck] = useState(INITIAL_CHECK_STATE);
  const [emailCheck, setEmailCheck] = useState(INITIAL_CHECK_STATE);

  // 이메일 형식 검증
  const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  // 입력값 변경 핸들러
  const handleInputChange = (field) => (e) => {
    setFormData(prev => ({ ...prev, [field]: e.target.value }));
  };

  // 아이디 입력 시 중복 확인 상태 초기화
  const handleUserIdChange = (e) => {
    const newValue = e.target.value;
    setFormData(prev => ({ ...prev, userId: newValue }));
    
    // 값이 변경될 때만 상태 초기화
    if (userIdCheck.checked) {
      setUserIdCheck(INITIAL_CHECK_STATE);
    }
  };

  // 이메일 입력 시 중복 확인 상태 초기화
  const handleEmailChange = (e) => {
    const newValue = e.target.value;
    setFormData(prev => ({ ...prev, email: newValue }));
    
    // 값이 변경될 때만 상태 초기화
    if (emailCheck.checked) {
      setEmailCheck(INITIAL_CHECK_STATE);
    }
  };

  // 중복 확인 결과 설정
  const setCheckResult = (setter, isValid, message) => {
    setter({
      checked: true,
      isValid,
      message
    });
  };

  // 아이디 중복 확인 함수
  const checkDuplicateUserId = async () => {
    if (!formData.userId.trim()) {
      alert(VALIDATION_MESSAGES.USER_ID_REQUIRED);
      return;
    }
    
    try {
      const response = await checkUserIdDuplicate(formData.userId);
      
      if (response.data === 'duplicate') {
        setCheckResult(setUserIdCheck, false, VALIDATION_MESSAGES.USER_ID_DUPLICATE);
      } else {
        setCheckResult(setUserIdCheck, true, VALIDATION_MESSAGES.USER_ID_AVAILABLE);
      }
    } catch (error) {
      console.error('아이디 중복 확인 오류:', error);
      alert('아이디 중복 확인 중 오류가 발생했습니다.');
    }
  };

  // 이메일 중복 확인 함수
  const checkDuplicateEmail = async () => {
    if (!formData.email.trim()) {
      alert(VALIDATION_MESSAGES.EMAIL_REQUIRED);
      return;
    }
    
    if (!validateEmail(formData.email)) {
      alert(VALIDATION_MESSAGES.EMAIL_FORMAT_INVALID);
      return;
    }
    
    try {
      const response = await checkEmailDuplicate(formData.email);
      
      if (response.data === 'duplicate') {
        setCheckResult(setEmailCheck, false, VALIDATION_MESSAGES.EMAIL_DUPLICATE);
      } else {
        setCheckResult(setEmailCheck, true, VALIDATION_MESSAGES.EMAIL_AVAILABLE);
      }
    } catch (error) {
      console.error('이메일 중복 확인 오류:', error);
      
      if (error.response?.status === 404) {
        setCheckResult(setEmailCheck, true, VALIDATION_MESSAGES.EMAIL_API_NOT_IMPLEMENTED);
      } else {
        setCheckResult(setEmailCheck, false, '이메일 중복 확인 중 오류가 발생했습니다.');
      }
    }
  };

  // 폼 검증 함수
  const validateForm = () => {
    if (!userIdCheck.checked || !userIdCheck.isValid) {
      alert(VALIDATION_MESSAGES.USER_ID_CHECK_REQUIRED);
      return false;
    }
    
    if (!emailCheck.checked || !emailCheck.isValid) {
      alert(VALIDATION_MESSAGES.EMAIL_CHECK_REQUIRED);
      return false;
    }
    
    if (formData.password !== formData.confirmPassword) {
      alert(VALIDATION_MESSAGES.PASSWORD_MISMATCH);
      return false;
    }
    
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    setLoading(true);
    
    try {
      const { confirmPassword, ...userData } = formData;
      const response = await registerUser(userData);
      
      if (response.data === "회원가입 성공") {
        alert(VALIDATION_MESSAGES.SIGNUP_SUCCESS);
        onClose();
      } else {
        alert('회원가입 실패: ' + response.data);
      }
    } catch (error) {
      console.error('회원가입 오류:', error);
      alert(VALIDATION_MESSAGES.SIGNUP_ERROR);
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
            회원가입
          </Typography>
        </CardHeader>
        <form onSubmit={handleSubmit}>
          <CardBody className="flex flex-col gap-4">
            {/* 아이디 입력 및 중복 확인 */}
            <DuplicateCheckField
              type="text"
              label="아이디"
              value={formData.userId}
              onChange={handleUserIdChange}
              onCheck={checkDuplicateUserId}
              checkState={userIdCheck}
              required={true}
            />
            
            <Input
              type="text"
              label="이름"
              size="lg"
              value={formData.name}
              onChange={handleInputChange('name')}
              required
            />
            
            {/* 이메일 입력 및 중복 확인 */}
            <DuplicateCheckField
              type="email"
              label="이메일"
              value={formData.email}
              onChange={handleEmailChange}
              onCheck={checkDuplicateEmail}
              checkState={emailCheck}
              required={true}
            />
            
            <Input
              type="tel"
              label="전화번호"
              size="lg"
              value={formData.phone}
              onChange={handleInputChange('phone')}
            />
            <Input
              type="password"
              label="비밀번호"
              size="lg"
              value={formData.password}
              onChange={handleInputChange('password')}
              required
            />
            <Input
              type="password"
              label="비밀번호 확인"
              size="lg"
              value={formData.confirmPassword}
              onChange={handleInputChange('confirmPassword')}
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
              {loading ? '가입 중...' : '회원가입'}
            </Button>
            <Typography variant="small" className="mt-4 flex justify-center">
              이미 계정이 있으신가요?{" "}
              <span className="ml-1 font-bold text-blue-500 cursor-pointer" onClick={onSwitchToLogin}>
                로그인
              </span>
            </Typography>
          </CardFooter>
        </form>
      </Card>

  );
};

export default Signup;