import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from "@material-tailwind/react";

// 메인화면의 헤더, 메인, 푸터
import Header from './components/Header';
import MainSection from './components/MainSection';
import Footer from './components/Footer';

import Signup from './components/SignupLogin/Signup.jsx';
import Login from './components/SignupLogin/Login.jsx';


// 보호견 관리 메뉴
import PetList from './components/PetManagement/PetList';
import ChatbotSearch from './components/PetManagement/ChatbotSearch';

// 입양 상담 메뉴
import ChatConsult from './components/AdoptConsulting/ChatConsult';

// 보호된 라우트 컴포넌트
import ProtectedRoute from './components/common/ProtectedRoute';

// AI 수의사 메뉴
import SkinCheck from './components/AIVet/SkinCheck';

// 유기견 위치 수색 메뉴
import SearchPetPage from './components/SearchPet/SearchPetPage';
import MissingPetInfo from './components/SearchPet/MissingPetInfo';
import ShowMap from './components/SearchPet/ShowMap';
import SearchResult from './components/SearchPet/SearchResult';
import SearchChatbotModal from './components/SearchPet/SearchChatbotModal';

// 강아지 유사도 검색 시스템 (우리가 만든 것)
import DogSimilaritySearch from './DogSimilaritySearch';

// 마이페이지 메뉴
import EditInfo from './components/Mypage/EditInfo'
import ChangePassword from './components/Mypage/ChangePassword';

// 관리자 메뉴
import AdminNotice from './components/Admin/Notice';
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
  return (
    <ThemeProvider>
      <Router>
        <Header /> {/* 모든 페이지에 헤더 고정 */}
        <Layout>
          <Routes>
            {/* 메인 페이지 */}
            <Route path="/" element={<MainSection />} />

            {/* 회원가입 페이지 */}
            <Route path="/signup" element={<Signup />} />
            
            {/* 로그인 페이지 */}
            <Route path="/login" element={<Login />} />

            {/* 유기견 관리 페이지 */}
            <Route path="/pet/list" element={<PetList />} />
            <Route path="/pet/chatbot" element={<ChatbotSearch />} />

            {/* 입양 상담 페이지 - 로그인 필요 */}
            <Route 
              path="/consult/chat" 
              element={
                <ProtectedRoute>
                  <ChatConsult />
                </ProtectedRoute>
              } 
            />

            {/* AI 수의사 페이지 */}
            <Route path="/vet/skin" element={<SkinCheck />} />

            {/* 유기견 위치 수색 페이지 */}
            <Route path="/search/petsearchpage" element={<SearchPetPage />} />
            <Route path="/search/missingpetinfo" element={<MissingPetInfo />} />
            <Route path="/search/showmap" element={<ShowMap />} />
            <Route path="/search/heatmap" element={<SearchResult />} />
            <Route path="/search/chatbot" element={<SearchChatbotModal />} />

            {/* 강아지 유사도 검색 시스템 (우리가 추가한 것) */}
            <Route path="/dog/similarity" element={<DogSimilaritySearch />} />

            {/* 마이페이지 */}
            <Route path="/mypage/edit" element={<EditInfo />} />
            <Route path="/mypage/password" element={<ChangePassword />} />

            {/* 관리자 페이지 */}
            <Route path="/admin/petregister" element={<AdminPetRegister />} />
            <Route path="/admin/petmanagement" element={<PetManagement />} />
            <Route path="/admin/notice" element={<AdminNotice />} />
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
      </Router>
    </ThemeProvider>
  );
}

export default App;