import React, { useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from "@material-tailwind/react";

// 공통코드 Context Provider
import { CommonCodeProvider } from './contexts/CommonCodeContext';

// 메인화면의 헤더, 메인, 푸터
import Header from './components/Header';
import MainSection from './components/MainSection';
import DrawerAccordionMenu from './components/DrawerAccordionMenu';
import { NODE_ENV } from './config/env';
import Footer from './components/Footer';

import Signup from './components/SignupLogin/Signup.jsx';
import Login from './components/SignupLogin/Login.jsx';


// 보호견 관리 메뉴
import PetList from './components/PetManagement/PetList';
import ChatbotSearch from './components/PetManagement/ChatbotSearch';

// 입양 상담 메뉴
import ChatConsult from './components/AdoptConsulting/ChatConsult.jsx';

// 보호된 라우트 컴포넌트 (현재 사용 안함 - 로그인 체크 제거)
// import ProtectedRoute from './components/auth/ProtectedRoute';
// import LocalOnlyRoute from './components/auth/LocalOnlyRoute';
// import AdminRoute from './components/auth/AdminRoute';

// AI 수의사 메뉴
import SkinCheck from './components/AIVet/SkinCheck';

// 유기견 위치 수색 메뉴
import SearchPetPage from './components/SearchPet/SearchPetPage';
import MissingPetInfo from './components/SearchPet/MissingPetInfo';
import ShowMap from './components/SearchPet/ShowMap';
import SearchResult from './components/SearchPet/SearchResult';
import SearchChatbotModal from './components/SearchPet/SearchChatbotModal';

// 강아지 유사도 검색 시스템
import DogSimilaritySearch from './components/PetManagement/DogSimilaritySearch';
import SearchedDogList from './components/PetManagement/SearchedDogList';
import DogDetailView from './components/PetManagement/DogDetailView';
import SearchMyDog from './components/PetManagement/SearchMyDog';

// 마이페이지 메뉴
import EditInfo from './components/Mypage/EditInfo'
import ChangePassword from './components/Mypage/ChangePassword';

// 관리자 메뉴
import AdminNotice from './components/Admin/Notice';
import AdminNoticeGetNewInfo from './components/Admin/NoticeGetNewInfo';
import AdminNoticeManagement from './components/Admin/NoticeManagement';
import AdminPetRegister from './components/Admin/PetRegister';
import AdminCommonCode from './components/Admin/CommonCode';
import AdminUserPermission from './components/Admin/UserPermission';
import PetManagement from './components/Admin/PetManagement';

// DB 연결 테스트
import DBTest from './components/DBTest';
import DBLLMTest from './components/DBLLMTest';

// 보호견 좌표 업데이트
import AnimalCoordinateUpdater from './components/AnimalCoordinateUpdater';

// 소셜 로그인 콜백 페이지
import SocialLoginCallback from './pages/SocialLoginCallback';


// 레이아웃 컴포넌트
const Layout = ({ children }) => {
  return (
    <div className="App min-h-screen bg-gray-50">
      {children}
      <Footer />
    </div>
  );
};


function App() {
  // Drawer 상태를 App에서 관리
  const [isDrawerOpen, setIsDrawerOpen] = React.useState(false);
  const drawerWidth = 256; // px, w-64

  // 회원가입/로그인/Colors 모달 상태 및 로그인 상태 관리
  const [openSignup, setOpenSignup] = React.useState(false);
  const [openLogin, setOpenLogin] = React.useState(false);
  const [isPubView, setIsPubView] = React.useState(false);
  const [isLoggedIn, setIsLoggedIn] = React.useState(false);
  const [userId, setUserId] = React.useState('');
  const [userName, setUserName] = React.useState('');

  // 로그인 상태 확인 (초기화)
  useEffect(() => {
    const loggedIn = localStorage.getItem('isLoggedIn') === 'true';
    const storedUserId = localStorage.getItem('userId') || '';
    const storedUserName = localStorage.getItem('name') || '';
    setIsLoggedIn(loggedIn);
    setUserId(storedUserId);
    setUserName(storedUserName);
  }, []);

  // 로그아웃 함수
  const handleLogout = () => {
    localStorage.removeItem('isLoggedIn');
    localStorage.removeItem('userId');
    localStorage.removeItem('token');
    localStorage.removeItem('name');
    localStorage.removeItem('role'); // ✅ role도 제거해야 함
    setIsLoggedIn(false);
    setUserId('');
    setUserName('');
    alert('로그아웃되었습니다.');
    window.location.replace('/');
  };

  // 멀티탭 로그아웃 동기화
  useEffect(() => {
    const syncLogout = (event) => {
      if (event.key === 'logout') {
        window.location.replace('/');
      }
    };
    window.addEventListener('storage', syncLogout);
    return () => window.removeEventListener('storage', syncLogout);
  }, []);

  // 윈도우 크기 변경 시 Drawer 자동 닫힘 (lg 이상)
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 1024) setIsDrawerOpen(false);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // 공통: 햄버거/Drawer가 보이는지 감지 (1024px 미만)
  const [showHamburger, setShowHamburger] = React.useState(false);
  useEffect(() => {
    function handleResize() {
      const navBarVisible = window.innerWidth >= 1024;
      setShowHamburger(!navBarVisible);
    }
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <ThemeProvider>

      <Router>
        {/* Header에 햄버거 클릭 핸들러 및 컨트롤 상태 전달 */}
        <Header
          onOpenDrawer={() => setIsDrawerOpen(true)}
          isLoggedIn={isLoggedIn}
          userId={userId}
          userName={userName}
          onOpenSignup={() => setOpenSignup(true)}
          onOpenLogin={() => setOpenLogin(true)}
          onLogout={handleLogout}
          onOpenColors={() => setIsPubView(true)}
          showRightControls={!showHamburger}
        />
        {/* Drawer(오른쪽 푸시) 메뉴: 모든 페이지에서 전역 렌더링 */}
        {isDrawerOpen && (
          <div
            className="fixed top-0 right-0 w-64 max-w-[80vw] h-full bg-white shadow-xl flex flex-col animate-slideInDrawer z-[99999] overflow-y-auto"
            style={{ minWidth: drawerWidth, transition: 'all 0.22s cubic-bezier(0.4,0,0.2,1)' }}
          >
            {/* 햄버거 버튼 (닫기용) */}
            <button
              onClick={() => setIsDrawerOpen(false)}
              aria-label="메뉴 닫기"
              className="flex items-center gap-2 px-4 py-4 text-blue-900 text-lg font-extrabold focus:outline-none hover:text-blue-700"
              style={{ alignSelf: 'flex-start', background: 'none', border: 'none' }}
            >
              <svg width="28" height="28" fill="none" stroke="currentColor" strokeWidth="2.2" viewBox="0 0 24 24" className="mr-1"><path strokeLinecap="round" strokeLinejoin="round" d="M4 7h16M4 12h16M4 17h16"/></svg>
              전체메뉴
            </button>
            {/* 네비게이션 메뉴: 아코디언(토글) 방식 */}
            <div className="px-4 pb-6 pt-2 flex-1">
              <DrawerAccordionMenu 
                onOpenSignup={() => setOpenSignup(true)}
                onOpenLogin={() => setOpenLogin(true)}
                isLoggedIn={isLoggedIn}
                userName={userName}
                onLogout={handleLogout}
              />
            </div>
          </div>
        )}
        <Layout>
          {/* 회원가입/로그인 모달 렌더링 */}
          {openSignup && <Signup onClose={() => setOpenSignup(false)} />}
          {openLogin && <Login onClose={() => setOpenLogin(false)} />}
          <Routes>
            {/* 메인 페이지 - Drawer 상태와 컨트롤 상태를 prop으로 전달 */}
            <Route
              path="/"
              element={
                <MainSection
                  isDrawerOpen={isDrawerOpen}
                  setIsDrawerOpen={setIsDrawerOpen}
                  drawerWidth={drawerWidth}
                  showRightControlsInDrawer={showHamburger}
                  isLoggedIn={isLoggedIn}
                  userId={userId}
                  userName={userName}
                  NODE_ENV={NODE_ENV}
                  onOpenSignup={() => setOpenSignup(true)}
                  onOpenLogin={() => setOpenLogin(true)}
                  onLogout={handleLogout}
                  onOpenColors={() => setIsPubView(true)}
                />
              }
            />

            {/* 유기견 관리 페이지 */}
            <Route path="/pet/list" element={<PetList />} />
            <Route path="/pet/search" element={<DogSimilaritySearch />} />

            {/* 입양 상담 페이지 */}
            <Route path="/consult/chat" element={<ChatConsult />} />

            {/* AI 수의사 페이지 */}
            <Route path="/vet/skin" element={<SkinCheck />} />

            {/* 유기견 위치 수색 페이지 */}
            <Route path="/search/petsearchpage" element={<SearchPetPage />} />
            <Route path="/search/missingpetinfo" element={<MissingPetInfo />} />
            <Route path="/search/showmap" element={<ShowMap />} />
            <Route path="/search/heatmap" element={<SearchResult />} />
            <Route path="/search/chatbot" element={<SearchChatbotModal />} />

            {/* 마이페이지 */}
            <Route path="/mypage/edit" element={<EditInfo />} />
            <Route path="/mypage/password" element={<ChangePassword />} />

            {/* 관리자 페이지 */}
            <Route path="/admin/petregister" element={<AdminPetRegister />} />
            <Route path="/admin/petmanagement" element={<PetManagement />} />
            <Route path="/admin/notice" element={<AdminNotice />} />
            <Route path="/admin/noticenewinfo" element={<AdminNoticeGetNewInfo />} />
            <Route path="/admin/noticemanagement" element={<AdminNoticeManagement />} />
            <Route path="/admin/commoncode" element={<AdminCommonCode />} />
            <Route path="/admin/userpermission" element={<AdminUserPermission />} />

            {/* DB 연결 테스트 페이지 */}
            <Route path='/dbtest' element={<DBTest />} />
            <Route path='/dbllmtest' element={<DBLLMTest />} />

            {/* 보호견 좌표 업데이트 페이지 */}
            <Route path='/animalcoord' element={<AnimalCoordinateUpdater />} />

            {/* 소셜 로그인 콜백 */}
            <Route path="/social/callback" element={<SocialLoginCallback />} />
          </Routes>
        </Layout>
        {/* 슬라이드 애니메이션 (Tailwind 커스텀) */}
        <style>{`
        @keyframes slideInDrawer {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
        .animate-slideInDrawer {
          animation: slideInDrawer 0.22s cubic-bezier(0.4,0,0.2,1);
        }
        `}</style>
      </Router>
    </ThemeProvider>
  );
}

export default App;