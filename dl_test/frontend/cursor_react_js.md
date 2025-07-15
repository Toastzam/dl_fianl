# React.js 스캐폴딩 소스 생성 요청
_Exported on 6/16/2025 at 23:19:04 GMT+9 from Cursor (0.50.5)_

---

**User**

React.js 스캐폴딩 소스를 생성해줘.
1)  www.animal.go.kr/front/index.do 같은 레이아웃으로 구성
   (상단 메뉴바, 중간에 배너 및 공지사항, 하단에 관련 사이트 링크 배너 슬라이드)
2) Tailwind CSS + daisyui 테마로 생성해줘
@import "tailwindcss";
@plugin "daisyui" {
  themes: light --default, dark --prefersdark, cupcake;
}

---

**Cursor**

네, React.js 스캐폴딩 소스를 생성해드리겠습니다. 먼저 현재 워크스페이스를 확인하고 해당 웹사이트의 레이아웃을 참고해서 프로젝트를 구성하겠습니다.

이제 React.js 프로젝트를 생성하고 www.animal.go.kr과 유사한 레이아웃으로 구성하겠습니다. 먼저 Vite를 사용해서 React 프로젝트를 생성하겠습니다.

이제 프로젝트 디렉토리로 이동하고 필요한 패키지들을 설치하겠습니다.

이제 Tailwind CSS와 daisyUI를 설치하겠습니다.

이제 Tailwind CSS 설정 파일을 생성하겠습니다.

수동으로 Tailwind CSS 설정 파일들을 생성하겠습니다.

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [require("daisyui")],
  daisyui: {
    themes: ["light", "dark", "cupcake"],
  },
}
```

```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

이제 CSS 파일을 수정하여 Tailwind CSS를 설정하겠습니다.

```css
@import "tailwindcss";
@plugin "daisyui" {
  themes: light --default, dark --prefersdark, cupcake;
}

@tailwind base;
@tailwind components;
@tailwind utilities;
```

이제 www.animal.go.kr과 유사한 레이아웃으로 메인 컴포넌트들을 생성하겠습니다. 먼저 Header 컴포넌트를 만들겠습니다.

```javascriptreact
import React, { useState } from 'react';

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const menuItems = [
    {
      title: '반려동물 등록 조회',
      subItems: [
        { name: '동물등록제도 안내', link: '#' },
        { name: '동물등록 대행기관 조회', link: '#' },
        { name: '동물등록확인', link: '#' },
        { name: '동물등록증 출력', link: '#' },
      ]
    },
    {
      title: '반려동물 분실·구조',
      subItems: [
        { name: '반려동물 분실 신고', link: '#' },
        { name: '보호센터 보호동물', link: '#' },
        { name: '구조 대처 요령', link: '#' },
      ]
    },
    {
      title: '반려동물 입양',
      subItems: [
        { name: '입양 안내', link: '#' },
        { name: '입양대상 동물', link: '#' },
        { name: '보호종료 동물', link: '#' },
        { name: '국가봉사동물 입양', link: '#' },
      ]
    },
    {
      title: '동물복지 인증',
      subItems: [
        { name: '동물복지 개요', link: '#' },
        { name: '동물복지 축산인증제', link: '#' },
        { name: '인증절차', link: '#' },
        { name: '동물복지축산농장', link: '#' },
      ]
    },
    {
      title: '알림·자료',
      subItems: [
        { name: '공지사항', link: '#' },
        { name: '홍보자료', link: '#' },
        { name: '실태조사', link: '#' },
        { name: '동물보호센터 안내', link: '#' },
      ]
    }
  ];

  return (
    <header className="bg-white shadow-lg">
      {/* 상단 유틸리티 바 */}
      <div className="bg-base-200">
        <div className="container mx-auto px-4 py-2">
          <div className="flex justify-between items-center text-sm">
            <div className="flex space-x-4">
              <span>동물실험윤리위원회</span>
              <span>동물사랑사진전</span>
              <span>업무시스템</span>
            </div>
            <div className="flex space-x-4">
              <span>회원가입</span>
              <span>로그인</span>
              <span>원격지원</span>
            </div>
          </div>
        </div>
      </div>

      {/* 메인 헤더 */}
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* 로고 */}
          <div className="flex items-center">
            <h1 className="text-2xl font-bold text-primary">
              국가동물보호정보시스템
            </h1>
          </div>

          {/* 검색 바 */}
          <div className="flex items-center space-x-4">
            <div className="form-control">
              <div className="input-group">
                <input 
                  type="text" 
                  placeholder="검색어를 입력해주세요" 
                  className="input input-bordered w-64" 
                />
                <button className="btn btn-primary">검색</button>
              </div>
            </div>
            
            {/* 모바일 메뉴 버튼 */}
            <button 
              className="btn btn-ghost lg:hidden"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>

        {/* 네비게이션 메뉴 */}
        <nav className={`${isMenuOpen ? 'block' : 'hidden'} lg:block mt-4`}>
          <div className="navbar bg-primary text-primary-content rounded-lg">
            <div className="navbar-start lg:hidden">
              <button 
                className="btn btn-ghost"
                onClick={() => setIsMenuOpen(!isMenuOpen)}
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
            </div>
            
            <div className="navbar-center lg:navbar-start">
              <ul className="menu menu-horizontal px-1 hidden lg:flex">
                {menuItems.map((item, index) => (
                  <li key={index} className="dropdown dropdown-hover">
                    <label tabIndex={0} className="btn btn-ghost">
                      {item.title}
                      <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </label>
                    <ul tabIndex={0} className="dropdown-content z-[1] menu p-2 shadow bg-base-100 rounded-box w-64 text-base-content">
                      {item.subItems.map((subItem, subIndex) => (
                        <li key={subIndex}>
                          <a href={subItem.link} className="hover:bg-primary hover:text-primary-content">
                            {subItem.name}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* 모바일 메뉴 */}
          <div className={`${isMenuOpen ? 'block' : 'hidden'} lg:hidden mt-4`}>
            <div className="bg-base-100 rounded-lg shadow-lg">
              {menuItems.map((item, index) => (
                <div key={index} className="collapse collapse-arrow">
                  <input type="checkbox" />
                  <div className="collapse-title text-lg font-medium">
                    {item.title}
                  </div>
                  <div className="collapse-content">
                    <ul className="menu">
                      {item.subItems.map((subItem, subIndex) => (
                        <li key={subIndex}>
                          <a href={subItem.link}>{subItem.name}</a>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </nav>
      </div>
    </header>
  );
};

export default Header;
```

이제 메인 배너와 공지사항 영역을 포함한 MainSection 컴포넌트를 생성하겠습니다.

```javascriptreact
import React, { useState, useEffect } from 'react';

const MainSection = () => {
  const [currentSlide, setCurrentSlide] = useState(0);

  const banners = [
    {
      id: 1,
      title: "2025 동물등록 자진신고 및 집중단속 기간 운영 안내",
      subtitle: "반려견 등록은 의무입니다. 자진신고 기간에 등록하면 과태료가 면제됩니다.",
      detail: "1차 자진신고: 5월 1일 ~ 6월 30일 / 집중단속: 7월 1일 ~ 7월 31일",
      bgColor: "bg-gradient-to-r from-blue-500 to-blue-600"
    },
    {
      id: 2,
      title: "맹견사육 허가제도 안내",
      subtitle: "2024년 4월 27일부터 맹견사육허가제가 시행됩니다.",
      detail: "동물보호법 개정으로 맹견을 사육하는 소유자는 시·도지사로부터 사육허가를 받아야 합니다.",
      bgColor: "bg-gradient-to-r from-red-500 to-red-600"
    },
    {
      id: 3,
      title: "고양이 동물등록, 전국 어디서나 가능해요!",
      subtitle: "반려묘와 함께 동물병원을 방문해 마이크로칩을 삽입하세요.",
      detail: "보호자 정보와 반려동물 정보를 작성한 뒤, 시·군·구청에서 동물등록증을 수령할 수 있습니다.",
      bgColor: "bg-gradient-to-r from-green-500 to-green-600"
    }
  ];

  const quickMenuItems = [
    { title: '동물등록정보 연결 및 변경방법', icon: '🔗', link: '#' },
    { title: '동물등록 정보조회', icon: '🔍', link: '#' },
    { title: '변경신고 및 등록증출력', icon: '📄', link: '#' },
    { title: '입양대상 동물', icon: '🐕', link: '#' },
    { title: '보호센터 보호동물', icon: '🏠', link: '#' },
    { title: '반려동물 판매생산수입업 거래내역', icon: '📊', link: '#' }
  ];

  const notices = [
    {
      id: 1,
      title: "시스템 점검 안내 [05.07.(수) 18:00 ~ 22:00]",
      date: "2025.05.07",
      isNew: true,
      content: "국가동물보호정보시스템의 안정적 운영을 위해 시스템 점검을 실시합니다."
    },
    {
      id: 2,
      title: "공동소유자의 동물등록증 출력",
      date: "2025.03.25",
      isNew: true,
      content: "반려견 공동소유자의 동물등록증 출력시 공동소유자 2인의 이름과 연락처가 모두 인쇄됩니다."
    },
    {
      id: 3,
      title: "동물등록제도 안내 업데이트",
      date: "2025.03.20",
      isNew: false,
      content: "동물등록제도 관련 최신 정보를 확인하세요."
    }
  ];

  const protectedAnimals = [
    {
      id: 1,
      type: "믹스견",
      location: "성환읍 성월 4길26",
      features: "순함, 25년 3월생",
      date: "2025-06-16",
      image: "🐕"
    },
    {
      id: 2,
      type: "한국 고양이",
      location: "원종 287-25",
      features: "눈꼽, 허피스, 마름",
      date: "2025-06-16",
      image: "🐱"
    },
    {
      id: 3,
      type: "비숑 프리제",
      location: "봉담읍 수영리 인근",
      features: "순하고 겁이 조금 있음",
      date: "2025-06-16",
      image: "🐕"
    }
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % banners.length);
    }, 5000);

    return () => clearInterval(interval);
  }, [banners.length]);

  return (
    <main className="min-h-screen bg-base-100">
      {/* 메인 배너 슬라이드 */}
      <section className="relative h-96 overflow-hidden">
        <div className="carousel w-full h-full">
          {banners.map((banner, index) => (
            <div
              key={banner.id}
              className={`carousel-item w-full h-full ${banner.bgColor} text-white flex items-center justify-center ${
                index === currentSlide ? 'block' : 'hidden'
              }`}
            >
              <div className="container mx-auto px-4 text-center">
                <h2 className="text-4xl font-bold mb-4">{banner.title}</h2>
                <p className="text-xl mb-2">{banner.subtitle}</p>
                <p className="text-lg opacity-90">{banner.detail}</p>
              </div>
            </div>
          ))}
        </div>
        
        {/* 슬라이드 인디케이터 */}
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-2">
          {banners.map((_, index) => (
            <button
              key={index}
              className={`w-3 h-3 rounded-full ${
                index === currentSlide ? 'bg-white' : 'bg-white bg-opacity-50'
              }`}
              onClick={() => setCurrentSlide(index)}
            />
          ))}
        </div>
      </section>

      <div className="container mx-auto px-4 py-8">
        {/* 퀵메뉴 */}
        <section className="mb-12">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {quickMenuItems.map((item, index) => (
              <div key={index} className="card bg-base-200 shadow-lg hover:shadow-xl transition-shadow cursor-pointer">
                <div className="card-body items-center text-center p-4">
                  <div className="text-3xl mb-2">{item.icon}</div>
                  <h3 className="text-sm font-medium">{item.title}</h3>
                </div>
              </div>
            ))}
          </div>
        </section>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* 공지사항 */}
          <section className="lg:col-span-2">
            <div className="card bg-base-200 shadow-lg">
              <div className="card-body">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="card-title text-2xl">공지사항</h2>
                  <button className="btn btn-primary btn-sm">더보기</button>
                </div>
                <div className="space-y-4">
                  {notices.map((notice) => (
                    <div key={notice.id} className="border-b border-base-300 pb-4 last:border-b-0">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <h3 className="font-medium hover:text-primary cursor-pointer">
                            {notice.title}
                            {notice.isNew && <span className="badge badge-error badge-sm ml-2">NEW</span>}
                          </h3>
                          <p className="text-sm text-base-content opacity-70 mt-1">{notice.content}</p>
                        </div>
                        <span className="text-sm text-base-content opacity-60 ml-4">{notice.date}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>

          {/* 동물등록제 안내 */}
          <section>
            <div className="card bg-primary text-primary-content shadow-lg">
              <div className="card-body">
                <h2 className="card-title">동물등록은 왜 해야하나요?</h2>
                <p className="text-sm opacity-90 mt-2">
                  등록대상동물의 소유자는 동물의 보호와 유실·유기 방지 등을 위하여 
                  2개월령 이상의 개를 소유한 사람은 반드시 동물등록을 해야합니다.
                </p>
                <div className="card-actions justify-end mt-4">
                  <button className="btn btn-primary-content btn-sm">동물등록제도 소개</button>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* 국가봉사동물 입양정보 */}
        <section className="mt-12">
          <div className="card bg-base-200 shadow-lg">
            <div className="card-body">
              <h2 className="card-title text-2xl mb-4">국가봉사동물 입양정보</h2>
              <p className="text-base-content opacity-80 mb-6">
                국가와 국민의 안전을 위해 헌신한 국가봉사동물에게 반려동물로써 안락한 삶을 누릴 수 있도록 입양을 지원합니다.
              </p>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {protectedAnimals.map((animal) => (
                  <div key={animal.id} className="card bg-base-100 shadow-md">
                    <div className="card-body p-4">
                      <div className="flex items-center mb-2">
                        <span className="text-2xl mr-2">{animal.image}</span>
                        <div>
                          <h3 className="font-medium">{animal.type}</h3>
                          <p className="text-sm opacity-60">{animal.date}</p>
                        </div>
                      </div>
                      <p className="text-sm"><span className="font-medium">발견장소:</span> {animal.location}</p>
                      <p className="text-sm"><span className="font-medium">특징:</span> {animal.features}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* 지역 동물병원·보호센터 */}
        <section className="mt-12">
          <div className="card bg-base-200 shadow-lg">
            <div className="card-body">
              <h2 className="card-title text-2xl mb-4">우리지역 동물병원·동물보호센터</h2>
              <div className="grid grid-cols-4 lg:grid-cols-8 gap-2 mb-4">
                {['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종'].map((city) => (
                  <button key={city} className="btn btn-outline btn-sm">{city}</button>
                ))}
              </div>
              <div className="grid grid-cols-4 lg:grid-cols-9 gap-2 mb-6">
                {['경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'].map((province) => (
                  <button key={province} className="btn btn-outline btn-sm">{province}</button>
                ))}
              </div>
              <div className="stats stats-vertical lg:stats-horizontal shadow">
                <div className="stat">
                  <div className="stat-title">동물보호센터</div>
                  <div className="stat-value text-primary">256</div>
                  <div className="stat-desc">개소</div>
                </div>
                <div className="stat">
                  <div className="stat-title">동물병원</div>
                  <div className="stat-value text-secondary">1,234</div>
                  <div className="stat-desc">개소</div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
};

export default MainSection;
```

이제 하단에 관련 사이트 링크 배너 슬라이드를 포함한 Footer 컴포넌트를 생성하겠습니다.

```javascriptreact
import React, { useState, useEffect } from 'react';

const Footer = () => {
  const [currentPartnerSlide, setCurrentPartnerSlide] = useState(0);

  const partnerSites = [
    {
      id: 1,
      name: '농림축산검역본부',
      url: 'https://www.qia.go.kr',
      description: '동물검역 및 질병관리'
    },
    {
      id: 2,
      name: '농림축산식품부',
      url: 'https://www.mafra.go.kr',
      description: '농림축산식품 정책'
    },
    {
      id: 3,
      name: '공공데이터포털',
      url: 'https://www.data.go.kr',
      description: '공공데이터 개방'
    },
    {
      id: 4,
      name: '동물사랑배움터',
      url: '#',
      description: '동물교육 플랫폼'
    },
    {
      id: 5,
      name: '지방행정인허가데이터',
      url: '#',
      description: '지방행정 데이터 개방'
    }
  ];

  const contactInfo = {
    phone: '1577-0954',
    systemInquiry: '054-810-8626',
    address: '39660 경상북도 김천시 혁신8로 177(율곡동)',
    workingHours: '평일 09:00 ~ 18:00'
  };

  const footerLinks = [
    { name: '시스템소개', url: '#' },
    { name: '저작권 정책', url: '#' },
    { name: '이용안내', url: '#' },
    { name: '개인정보처리방침', url: '#' }
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentPartnerSlide((prev) => (prev + 1) % Math.ceil(partnerSites.length / 3));
    }, 4000);

    return () => clearInterval(interval);
  }, [partnerSites.length]);

  const getSlidesForIndex = (slideIndex) => {
    const startIndex = slideIndex * 3;
    return partnerSites.slice(startIndex, startIndex + 3);
  };

  return (
    <footer className="bg-base-300 mt-16">
      {/* 동물사랑배움터 배너 */}
      <section className="bg-primary text-primary-content py-8">
        <div className="container mx-auto px-4">
          <div className="flex flex-col lg:flex-row items-center justify-between">
            <div className="text-center lg:text-left mb-4 lg:mb-0">
              <h3 className="text-2xl font-bold mb-2">동물사랑배움터</h3>
              <p className="opacity-90">
                동물 관련 영업자, 명예감시원, 맹견소유자는 연1회 법정의무교육을 이수해야 합니다.
              </p>
            </div>
            <button className="btn btn-primary-content">동물사랑배움터 바로가기</button>
          </div>
        </div>
      </section>

      {/* 동물보호 상담센터 */}
      <section className="bg-accent text-accent-content py-6">
        <div className="container mx-auto px-4">
          <div className="flex flex-col lg:flex-row items-center justify-between">
            <div className="text-center lg:text-left mb-4 lg:mb-0">
              <h3 className="text-xl font-bold">동물보호 상담센터</h3>
              <div className="flex flex-col lg:flex-row lg:space-x-8 mt-2">
                <div>
                  <span className="font-semibold">상담전화:</span> {contactInfo.phone}
                </div>
                <div>
                  <span className="font-semibold">시스템 문의:</span> {contactInfo.systemInquiry}
                </div>
                <div>
                  <span className="font-semibold">상담시간:</span> {contactInfo.workingHours}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 관련기관 슬라이드 */}
      <section className="bg-base-200 py-8">
        <div className="container mx-auto px-4">
          <h3 className="text-xl font-bold text-center mb-6">관련기관</h3>
          <div className="relative">
            <div className="overflow-hidden">
              <div className="flex transition-transform duration-300 ease-in-out">
                {Array.from({ length: Math.ceil(partnerSites.length / 3) }).map((_, slideIndex) => (
                  <div
                    key={slideIndex}
                    className={`w-full flex-shrink-0 ${
                      slideIndex === currentPartnerSlide ? 'block' : 'hidden'
                    }`}
                  >
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {getSlidesForIndex(slideIndex).map((site) => (
                        <div key={site.id} className="card bg-base-100 shadow-md hover:shadow-lg transition-shadow">
                          <div className="card-body text-center p-4">
                            <h4 className="font-semibold text-lg">{site.name}</h4>
                            <p className="text-sm opacity-70 mt-2">{site.description}</p>
                            <div className="card-actions justify-center mt-4">
                              <button className="btn btn-primary btn-sm">방문하기</button>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* 슬라이드 네비게이션 */}
            <div className="flex justify-center mt-6 space-x-2">
              {Array.from({ length: Math.ceil(partnerSites.length / 3) }).map((_, index) => (
                <button
                  key={index}
                  className={`w-3 h-3 rounded-full ${
                    index === currentPartnerSlide ? 'bg-primary' : 'bg-base-300'
                  }`}
                  onClick={() => setCurrentPartnerSlide(index)}
                />
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* 하단 정보 */}
      <section className="bg-base-300 py-8">
        <div className="container mx-auto px-4">
          <div className="flex flex-col lg:flex-row justify-between items-start">
            {/* 링크 메뉴 */}
            <div className="mb-6 lg:mb-0">
              <nav className="flex flex-wrap gap-4 mb-4">
                {footerLinks.map((link, index) => (
                  <a
                    key={index}
                    href={link.url}
                    className="text-base-content hover:text-primary transition-colors"
                  >
                    {link.name}
                  </a>
                ))}
              </nav>
            </div>

            {/* 연락처 정보 */}
            <div className="text-sm text-base-content opacity-80">
              <p className="font-semibold mb-2">농림축산검역본부</p>
              <p className="mb-1">{contactInfo.address}</p>
              <p className="mb-1">전화번호: {contactInfo.phone} (시스템 문의 {contactInfo.systemInquiry})</p>
              <p className="mb-4">Copyright by Animal and Plant Quarantine Agency. All Rights Reserved.</p>
              
              {/* 웹 접근성 인증 마크 */}
              <div className="flex items-center space-x-2">
                <div className="badge badge-outline">
                  웹 접근성 우수사이트 인증마크
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </footer>
  );
};

export default Footer;
```

이제 메인 App.jsx 파일을 수정하여 모든 컴포넌트를 통합하겠습니다.

```javascriptreact
import React from 'react'
import Header from './components/Header'
import MainSection from './components/MainSection'
import Footer from './components/Footer'
import './App.css'

function App() {
  return (
    <div className="App min-h-screen bg-base-100">
      <Header />
      <MainSection />
      <Footer />
    </div>
  )
}

export default App
```

이제 components 디렉토리를 생성하고 필요한 디렉토리 구조를 만들겠습니다.

이제 package.json에 프로젝트 정보를 업데이트하고 README.md 파일을 생성하겠습니다.

```json
{
  "name": "animal-portal-app",
  "description": "국가동물보호정보시스템 스타일의 React.js 웹 애플리케이션 - Tailwind CSS + daisyUI 테마",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "lint": "eslint . --ext js,jsx --report-unused-disable-directives --max-warnings 0",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.4",
    "eslint": "^9.15.0",
    "eslint-plugin-react": "^7.37.2",
    "eslint-plugin-react-hooks": "^5.0.0",
    "eslint-plugin-react-refresh": "^0.4.14",
    "vite": "^6.0.1",
    "tailwindcss": "^3.4.16",
    "postcss": "^8.5.0",
    "autoprefixer": "^10.4.20",
    "@tailwindcss/typography": "^0.5.15",
    "daisyui": "^4.12.14"
  }
}
```

이제 README.md 파일을 생성하여 프로젝트 설명을 추가하겠습니다.

```markdown
# 국가동물보호정보시스템 스타일 웹 애플리케이션

이 프로젝트는 [www.animal.go.kr](https://www.animal.go.kr/front/index.do)와 유사한 레이아웃으로 구성된 React.js 웹 애플리케이션입니다.

## 주요 기능

- 📱 반응형 디자인 (모바일, 태블릿, 데스크톱)
- 🎨 Tailwind CSS + daisyUI 테마 적용
- 🔍 상단 네비게이션 메뉴 (드롭다운 서브메뉴 포함)
- 🖼️ 자동 슬라이드 메인 배너
- 📋 퀵메뉴 및 공지사항 섹션
- 🐕 국가봉사동물 입양정보 및 보호동물 현황
- 🏥 지역별 동물병원·보호센터 안내
- 🔗 관련기관 링크 배너 슬라이드
- ♿ 웹 접근성 고려

## 기술 스택

- **Frontend**: React.js 18
- **Build Tool**: Vite
- **Styling**: Tailwind CSS 3 + daisyUI 4
- **State Management**: React Hooks (useState, useEffect)

## 레이아웃 구성

### 1. Header (상단)
- 유틸리티 바 (회원가입, 로그인, 원격지원 등)
- 로고 및 검색 바
- 드롭다운 메뉴가 있는 네비게이션

### 2. Main Section (중간)
- 자동 슬라이드 메인 배너 (3개 슬라이드)
- 퀵메뉴 그리드 (6개 항목)
- 공지사항 섹션
- 동물등록제 안내 카드
- 국가봉사동물 입양정보
- 보호동물 현황
- 지역별 동물병원·보호센터 통계

### 3. Footer (하단)  
- 동물사랑배움터 배너
- 동물보호 상담센터 연락처
- 관련기관 슬라이드 배너
- 저작권 및 연락처 정보

## 설치 및 실행

### 1. 의존성 설치
```bash
npm install
```

### 2. 개발 서버 실행
```bash
npm run dev
```

### 3. 빌드
```bash
npm run build
```

### 4. 빌드 미리보기
```bash
npm run preview
```

## daisyUI 테마 설정

현재 적용된 테마:
- `light` (기본)
- `dark` (다크모드)
- `cupcake` (추가 테마)

테마는 `tailwind.config.js` 파일에서 수정할 수 있습니다.

## 컴포넌트 구조

```
src/
├── components/
│   ├── Header.jsx      # 상단 헤더 및 네비게이션
│   ├── MainSection.jsx # 메인 콘텐츠 섹션
│   └── Footer.jsx      # 하단 푸터 및 관련기관 슬라이드
├── App.jsx             # 메인 애플리케이션
├── main.jsx           # 엔트리 포인트
└── index.css          # 전역 스타일 및 Tailwind CSS
```

## 반응형 브레이크포인트

- **Mobile**: < 768px
- **Tablet**: 768px - 1024px  
- **Desktop**: > 1024px

## 커스터마이징

### 색상 테마 변경
`tailwind.config.js`의 daisyUI 테마 설정을 수정하세요.

### 콘텐츠 수정
각 컴포넌트 파일의 데이터 배열을 수정하여 콘텐츠를 변경할 수 있습니다.

### 슬라이드 설정
- 메인 배너: `MainSection.jsx`의 `banners` 배열
- 관련기관: `Footer.jsx`의 `partnerSites` 배열

## 라이센스

이 프로젝트는 학습 및 참고 목적으로 제작되었습니다.
```

이제 개발 서버를 실행하여 애플리케이션을 테스트해보겠습니다.

React.js 스캐폴딩 소스 생성이 완료되었습니다! 🎉

## 완성된 프로젝트 구조:

```
animal-portal-app/
├── src/
│   ├── components/
│   │   ├── Header.jsx      # 상단 메뉴바 및 네비게이션
│   │   ├── MainSection.jsx # 메인 배너, 공지사항, 퀵메뉴
│   │   └── Footer.jsx      # 관련 사이트 링크 배너 슬라이드
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css          # Tailwind CSS + daisyUI 설정
├── tailwind.config.js     # Tailwind 및 daisyUI 테마 설정
├── postcss.config.js
├── package.json
└── README.md
```

## 주요 구현 사항:

### ✅ 1. 상단 메뉴바
- 유틸리티 바 (회원가입, 로그인, 원격지원)
- 로고 및 검색 기능
- 드롭다운 서브메뉴가 있는 반응형 네비게이션
- 모바일용 햄버거 메뉴

### ✅ 2. 중간 배너 및 공지사항
- **자동 슬라이드 메인 배너** (3개 슬라이드, 5초 간격)
- **퀵메뉴 그리드** (6개 항목)
- **공지사항** 섹션 (NEW 배지 포함)
- **동물등록제** 안내 카드
- **국가봉사동물** 입양정보
- **보호동물** 현황 카드들
- **지역별 동물병원·보호센터** 통계

### ✅ 3. 하단 관련 사이트 링크 배너 슬라이드
- **동물사랑배움터** 배너
- **동물보호 상담센터** 연락처
- **관련기관** 자동 슬라이드 (농림축산검역본부, 농림축산식품부 등)
- **저작권 및 연락처** 정보

### ✅ 4. Tailwind CSS + daisyUI 테마
- `light`, `dark`, `cupcake` 테마 설정
- 완전 반응형 디자인
- daisyUI 컴포넌트 활용 (card, btn, navbar, carousel 등)

## 실행 방법:

개발 서버가 백그라운드에서 실행 중입니다. 브라우저에서 다음 주소로 접속하세요:

**http://localhost:5173**

## 추가 명령어:

```bash
# 프로젝트 디렉토리로 이동
cd animal-portal-app

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev

# 프로덕션 빌드
npm run build

# 빌드 미리보기
npm run preview
```

이제 www.animal.go.kr과 매우 유사한 레이아웃의 React.js 애플리케이션을 사용하실 수 있습니다. 모든 컴포넌트는 재사용 가능하며, 쉽게 커스터마이징할 수 있도록 구조화되어 있습니다.

```plaintext
animal-portal-app/
├── src/
│   ├── components/
│   │   ├── Header.jsx      # 상단 메뉴바 및 네비게이션
│   │   ├── MainSection.jsx # 메인 배너, 공지사항, 퀵메뉴
│   │   └── Footer.jsx      # 관련 사이트 링크 배너 슬라이드
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css          # Tailwind CSS + daisyUI 설정
├── tailwind.config.js     # Tailwind 및 daisyUI 테마 설정
├── postcss.config.js
├── package.json
└── README.md
```

```shellscript
# 프로젝트 디렉토리로 이동
cd animal-portal-app

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev

# 프로덕션 빌드
npm run build

# 빌드 미리보기
npm run preview
```

