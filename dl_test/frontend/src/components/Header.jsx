import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Navbar,
  Typography,
  Button,
  IconButton,
  Input,
  Menu,
  MenuHandler,
  MenuList,
  MenuItem,
  Dialog,
} from "@material-tailwind/react";
import {
  MagnifyingGlassIcon,
  ChevronDownIcon,
} from "@heroicons/react/24/outline";

import Signup from './SignupLogin/Signup.jsx';
import Login from './SignupLogin/Login.jsx';
import ColorPreview from './pub/ColorPreview.jsx';
import { NODE_ENV } from '@/config/env';
import { checkAuthStatus, getUserRole, clearAuthData } from '@/utils/auth';


const Header = (props) => {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [openMenu, setOpenMenu] = useState(null);
  const closeTimeout = useRef();
  const [openSignup, setOpenSignup] = useState(false);
  const [openLogin, setOpenLogin] = useState(false);
  const [isPubView, setIsPubView] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userId, setUserId] = useState('');
  const [userName, setUserName] = useState('');
  const [userRole, setUserRole] = useState(''); // 사용자 역할 상태 추가

  const navigate = useNavigate(); // ✅ 추가됨: 라우터 훅

  // 컴포넌트 마운트 시 로그인 상태 확인
  useEffect(() => {
    const validateAndSetAuth = async () => {
      // 토큰 유효성 검증
      const isValid = await checkAuthStatus();
      
      if (isValid) {
        // 유효한 토큰이 있는 경우
        const storedUserId = localStorage.getItem('userId') || '';
        const storedUserName = localStorage.getItem('name') || '';
        const role = await getUserRole();
        
        setIsLoggedIn(true);
        setUserId(storedUserId);
        setUserName(storedUserName);
        setUserRole(role);
      } else {
        // 유효하지 않은 토큰이면 상태 초기화
        setIsLoggedIn(false);
        setUserId('');
        setUserName('');
        setUserRole('');
      }
    };

    validateAndSetAuth();

    // localStorage 변경 감지를 위한 이벤트 리스너
    const handleStorageChange = async () => {
      const isValid = await checkAuthStatus();
      
      if (isValid) {
        const updatedUserId = localStorage.getItem('userId') || '';
        const updatedUserName = localStorage.getItem('name') || '';
        const updatedUserRole = await getUserRole();
        setIsLoggedIn(true);
        setUserId(updatedUserId);
        setUserName(updatedUserName);
        setUserRole(updatedUserRole);
      } else {
        setIsLoggedIn(false);
        setUserId('');
        setUserName('');
        setUserRole('');
      }
    };

    // 커스텀 로그아웃 이벤트 처리 (같은 탭에서 즉시 반영)
    const handleUserLogout = () => {
      console.log('[Header] 로그아웃 이벤트 감지');
      setIsLoggedIn(false);
      setUserId('');
      setUserName('');
      setUserRole('');
    };

    window.addEventListener('storage', handleStorageChange);
    window.addEventListener('userLogout', handleUserLogout);
    
    // 컴포넌트 언마운트 시 이벤트 리스너 제거
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('userLogout', handleUserLogout);
    };
  }, []);

  // 로그아웃 함수
  const handleLogout = () => {
    clearAuthData(); // 유틸리티 함수 사용
    setIsLoggedIn(false);
    setUserId('');
    setUserName('');
    setUserRole(''); // 역할 상태도 초기화
    
    // 멀티탭 동기화를 위한 storage 이벤트 발생
    localStorage.setItem('logout', Date.now().toString());
    localStorage.removeItem('logout');
    
    // 같은 탭에서도 즉시 반영되도록 커스텀 이벤트 발생
    window.dispatchEvent(new CustomEvent('userLogout'));
    
    alert('로그아웃되었습니다.');
    navigate('/');
  };

  const toggleIsNavOpen = () => setIsNavOpen((cur) => !cur);

  // 기본 메뉴 항목들
  const baseMenuItems = [
    {
      title: '보호견 정보 관리',
      subItems: [
        { name: '보호견 목록', link: '/pet/list' },
        { name: '보호견 AI 검색', link: '/pet/search' },
      ]
    },
    {
      title: '입양상담',
      subItems: [
        { name: '챗봇 상담', link: '/consult/chat' },
      ]
    },
    {
      title: 'AI 수의사',
      subItems: [
        { name: '피부 진단', link: '/vet/skin' },
      ]
    },
    {
      title: '보호견 위치 탐색',
      subItems: [
        { name: '보호견 발견 신고', link: '/search/petsearchpage' },
        { name: '보호견 구조 가이드', link: '/search/heatmap' },
      ]
    },
    {
      title: '마이페이지',
      subItems: [
        { name: '회원 정보수정', link: '/mypage/edit' },
        { name: '비밀번호 수정', link: '/mypage/password' },
      ]
    }
  ];

  // 관리자 메뉴 항목
  const adminMenuItems = {
    title: '관리자',
    subItems: [
      { name: '보호견 관리', link: '/admin/petManagement' },
      { name: '공지사항', link: '/admin/notice' },
      { name: '공통코드 관리', link: '/admin/commoncode' },
      { name: '사용자 권한 관리', link: '/admin/userpermission' },
    ]
  };

  // 사용자 역할에 따라 메뉴 항목 결정
  const menuItems = userRole === 'ADMIN' 
    ? [...baseMenuItems, adminMenuItems] 
    : baseMenuItems;

  // NavList에 ref를 추가하여 외부 클릭 감지
  const navListRef = useRef(null);

  // 외부 클릭 감지 useEffect 제거 (Material Tailwind Menu가 자체적으로 처리)

  const NavList = () => {
    // 딜레이 닫힘 적용
    const handleMouseEnter = (index) => {
      clearTimeout(closeTimeout.current);
      setOpenMenu(index);
    };
    const handleMouseLeave = () => {
      closeTimeout.current = setTimeout(() => setOpenMenu(null), 180);
    };
    return (
      <ul
        ref={navListRef}
        className="mt-0 mb-0 flex flex-col gap-2 lg:mb-0 lg:mt-0 lg:flex-row lg:items-stretch lg:justify-between w-full h-full"
      >
        {menuItems.map((item, index) => (
          <Menu key={index} open={openMenu === index}>
            <div
              onMouseEnter={() => handleMouseEnter(index)}
              onMouseLeave={handleMouseLeave}
              className="w-full h-full"
            >
              <MenuHandler>
                <Button
                  variant="text"
                  className="flex items-center justify-center gap-1 text-sm font-medium capitalize tracking-tight text-gray-800 hover:text-gray-600 w-full bg-amber-100 hover:bg-amber-200 transition-all duration-200 h-full rounded-none px-2"
                >
                  {item.title}
                  <ChevronDownIcon
                    strokeWidth={2.5}
                    className={`h-4 w-4 transition-transform ${openMenu === index ? "rotate-180" : ""}`}
                  />
                </Button>
              </MenuHandler>
              <MenuList
                className="min-w-[200px] bg-amber-50 border-amber-50 rounded-none shadow-lg"
                style={{
                  marginTop: '0',
                  borderTop: 'none',
                  borderTopLeftRadius: '0',
                  borderTopRightRadius: '0'
                }}
                onMouseEnter={() => handleMouseEnter(index)}
                onMouseLeave={handleMouseLeave}
              >
                {item.subItems.map((subItem, subIndex) => (
                  <MenuItem
                    key={subIndex}
                    className="flex items-center gap-2 hover:bg-amber-100 transition-colors duration-200"
                    onClick={(e) => {
                      e.stopPropagation();
                      setOpenMenu(null);
                      navigate(subItem.link);
                    }}
                  >
                    <Typography
                      variant="small"
                      className="font-normal text-gray-700 hover:text-gray-700 transition-colors duration-200"
                    >
                      {subItem.name}
                    </Typography>
                  </MenuItem>
                ))}
              </MenuList>
            </div>
          </Menu>
        ))}
      </ul>
    );
  };


  // Drawer 상태는 MainSection에서 관리, prop으로 제어

  return (
    <header className="bg-white shadow-lg w-full min-w-0 overflow-visible">
      <Navbar className="w-full min-w-0 mx-auto max-w-screen-2xl px-4 py-4 border-0 rounded-none shadow-none navbar-container">
        <div className="container mx-auto max-w-screen-2xl flex flex-wrap items-center justify-between min-w-0 w-full overflow-visible">
          {/* 로고 + 회원가입/로그인/Colors 우측 정렬 */}
          <div className="flex w-full items-center justify-between pr-2 lg:pr-4">
            <Typography
              as="a"
              href="/"
              onClick={(e) => {
                e.preventDefault();
                navigate('/');
              }}
              className="mr-4 cursor-pointer py-1.5 font-bold text-xl text-blue-600 hover:text-blue-700 transition-colors"
            >
              <div className="flex items-center gap-2 text-xl">
                <img src="/favicon.svg" alt="logo" className="w-8 h-8" />
                보호견 통합 케어 시스템
              </div>
            </Typography>
            <div className="flex items-center gap-6 ml-auto">
              {!isLoggedIn ? (
                <>
                  <Typography
                    variant="small"
                    className="text-blue-gray-600 hover:text-blue-500 cursor-pointer"
                    onClick={() => setOpenSignup(true)}
                  >
                    회원가입
                  </Typography>
                  <Typography
                    variant="small"
                    className="text-blue-gray-600 hover:text-blue-500 cursor-pointer"
                    onClick={() => setOpenLogin(true)}
                  >
                    로그인
                  </Typography>
                </>
              ) : (
                <div className="flex items-center gap-4">
                  <Typography
                    variant="small"
                    className="text-blue-gray-600"
                  >
                    {userName ? `${userName}님, 환영합니다!` : `${userId}님, 환영합니다!`}
                  </Typography>
                  <Button
                    variant="outlined"
                    className="text-blue-gray-600 border-blue-gray-300 hover:bg-blue-gray-100"
                    onClick={handleLogout}
                  >
                    로그아웃
                  </Button>
                </div>
              )}
              
              {/* 전체메뉴 버튼: 네비게이션 바가 사라지는 순간에만 보임 (JS로 감지) */}
              {(() => {
                const [showHamburger, setShowHamburger] = React.useState(false);
                React.useEffect(() => {
                  function handleResize() {
                    // Tailwind lg: 1024px 기준, 네비게이션 바는 lg:block, 그 미만은 hidden
                    // 즉, 1024px 미만(확대 포함)에서만 햄버거 버튼 보이게
                    // (Tailwind는 확대시 devicePixelRatio 반영 안함, 실제 px 기준)
                    const navBarVisible = window.innerWidth >= 1024;
                    setShowHamburger(!navBarVisible);
                  }
                  handleResize();
                  window.addEventListener('resize', handleResize);
                  return () => window.removeEventListener('resize', handleResize);
                }, []);
                if (!showHamburger) return null;
                return (
                  <button
                    type="button"
                    className="flex items-center gap-1 font-extrabold text-blue-900 text-base px-2 py-1 hover:text-blue-700 focus:outline-none"
                    style={{ letterSpacing: '-0.01em' }}
                    onClick={typeof props?.onOpenDrawer === 'function' ? props.onOpenDrawer : undefined}
                  >
                    <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2.2" viewBox="0 0 24 24" className="mr-1"><path strokeLinecap="round" strokeLinejoin="round" d="M4 7h16M4 12h16M4 17h16"/></svg>
                    {/* 전체메뉴 */}
                  </button>
                );
              })()}
            </div>
          </div>
        </div>

        {/* 네비게이션 메뉴 (노란색 바): 데스크탑(lg 이상)에서만 보임, 모바일/확대(lg 이하)에서는 숨김 */}
        <div className="w-full bg-amber-100 shadow-sm mt-2 hidden lg:block">
          <div className="w-full px-6 flex items-center justify-between">
            {/* 네비게이션 메뉴: 데스크탑에서만 보임 */}
            <div className="w-full flex justify-between">
              <NavList />
            </div>
          </div>
        </div>
      </Navbar>

      {/* Drawer/메인 flex-row는 MainSection에서 관리 */}

      {/* 회원가입 모달 */}
      <Dialog
        open={openSignup}
        handler={() => setOpenSignup(false)}
        size="sm"
        className="p-0 bg-transparent shadow-none"
      >
        <div className="flex items-center justify-center min-h-screen">
          <Signup 
            onClose={() => setOpenSignup(false)}
            onSwitchToLogin={() => { setOpenSignup(false); setOpenLogin(true); }}
          />
        </div>
      </Dialog>

      {/* 로그인 모달 */}
      <Dialog
        open={openLogin}
        handler={() => setOpenLogin(false)}
        size="sm"
        className="p-0 bg-transparent shadow-none"
      >
        <div className="flex items-center justify-center min-h-screen">
          <Login 
            onClose={() => setOpenLogin(false)}
            onSwitchToSignup={() => { setOpenLogin(false); setOpenSignup(true); }}
          />
        </div>
      </Dialog>

      {/* ColorChart 모달 */}
      <Dialog
        open={isPubView}
        handler={() => setIsPubView(false)}
        size="lg"
      >
        <div className="flex justify-center h-[90vh]">
          <ColorPreview onClose={() => setIsPubView(false)} />
        </div>
      </Dialog>

      {/* 슬라이드 애니메이션 (Tailwind 커스텀)는 MainSection에서 필요시 추가 */}
    </header>
  );
};

export default Header;