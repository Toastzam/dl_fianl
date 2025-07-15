import React from 'react';
import { useNavigate } from 'react-router-dom';

function DrawerAccordionMenu({ onOpenSignup, onOpenLogin, isLoggedIn, userName, onLogout }) {
  const [openIdx, setOpenIdx] = React.useState(null);
  const navigate = useNavigate();
  // 메뉴별 세부 항목과 경로 매핑
  const menuData = [
    {
      title: '보호견 정보 관리',
      subItems: [
        { label: '보호견 목록', path: '/pet/list' },
        { label: '보호견 AI 검색', path: '/pet/search' }
      ]
    },
    {
      title: '입양상담',
      subItems: [
        { label: '챗봇 상담', path: '/consult/chat' }
      ]
    },
    {
      title: 'AI 수의사',
      subItems: [
        { label: '피부 진단', path: '/vet/skin' }
      ]
    },
    {
      title: '보호견 위치 탐색',
      subItems: [
        { label: '보호견 발견 신고', path: '/search/petsearchpage' },
        { label: '보호견 구조 가이드', path: '/search/heatmap' }
      ]
    },
    {
      title: '마이페이지',
      subItems: [
        { label: '회원 정보수정', path: '/mypage/edit' },
        { label: '비밀번호 수정', path: '/mypage/password' }
      ]
    },
    {
      title: '관리자',
      subItems: [
        { label: '보호견 관리', path: '/admin/petmanagement' },
        { label: '공지사항', path: '/admin/notice' },
        { label: '공통코드 관리', path: '/admin/commoncode' },
        { label: '사용자 권한 관리', path: '/admin/userpermission' }
      ]
    }
  ];
  return (
    <nav className="flex-1 overflow-y-auto">
      <ul className="flex flex-col gap-2">
        {menuData.map((menu, idx) => (
          <li key={menu.title}>
            <button
              type="button"
              className="w-full flex items-center justify-between font-bold text-blue-800 text-base mb-1 px-1 py-1 focus:outline-none"
              onClick={() => setOpenIdx(openIdx === idx ? null : idx)}
            >
              <span>{menu.title}</span>
              <svg className={`w-4 h-4 ml-2 transition-transform ${openIdx === idx ? 'rotate-90' : ''}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" /></svg>
            </button>
            {openIdx === idx && (
              <ul className="pl-3 flex flex-col gap-1 mb-2">
                {menu.subItems.map((sub, subIdx) => (
                  <li key={subIdx}>
                    <button
                      className="text-gray-500 text-sm text-left w-full hover:text-blue-600 hover:underline transition"
                      style={{ padding: '0.18rem 0.2rem' }}
                      onClick={() => { navigate(sub.path); }}
                    >
                      {sub.label}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
      {/* 하단: 로그인/로그아웃 버튼 및 환영 메시지 */}
      <div className="mt-8 flex flex-col gap-2 items-stretch">
        {isLoggedIn ? (
          <>
            <button
              className="w-full py-2 rounded bg-gray-200 text-blue-800 font-semibold hover:bg-gray-300 transition"
              onClick={onLogout}
            >
              로그아웃
            </button>
            <div className="mt-2 text-center">
              <span className="block text-blue-700 font-bold text-base">{userName}님, 환영합니다!</span>
            </div>
          </>
        ) : (
          <>
            <button
              className="w-full py-2 rounded bg-blue-600 text-white font-semibold hover:bg-blue-700 transition"
              onClick={onOpenSignup}
            >
              회원가입
            </button>
            <button
              className="w-full py-2 rounded bg-gray-200 text-blue-800 font-semibold hover:bg-gray-300 transition"
              onClick={onOpenLogin}
            >
              로그인
            </button>
          </>
        )}
      </div>
    </nav>
  );
}

export default DrawerAccordionMenu;
