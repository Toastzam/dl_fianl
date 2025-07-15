# 국가동물보호정보시스템 스타일 웹 애플리케이션 (Material Tailwind)

이 프로젝트는 [www.animal.go.kr](https://www.animal.go.kr/front/index.do)와 유사한 레이아웃으로 구성된 React.js 웹 애플리케이션입니다. **Material Tailwind**를 사용하여 Material Design 스타일로 구현되었습니다.

## 주요 기능

- 📱 **반응형 디자인** (모바일, 태블릿, 데스크톱)
- 🎨 **Material Tailwind** 컴포넌트 시스템 적용
- 🎭 **Material Design** 디자인 언어 구현
- 🔍 **상단 네비게이션 메뉴** (드롭다운 서브메뉴 포함)
- 🖼️ **자동 슬라이드 메인 배너** (Material Carousel)
- 📋 **퀵메뉴 및 공지사항 섹션** (Material Cards)
- 🐕 **국가봉사동물 입양정보** 및 보호동물 현황
- 🏥 **지역별 동물병원·보호센터** 안내
- 🔗 **관련기관 링크 배너 슬라이드**
- ♿ **웹 접근성** 고려
- 🌊 **Material Design 애니메이션** 및 인터랙션

## 기술 스택

- **Frontend**: React.js 18
- **Build Tool**: Vite
- **UI Library**: Material Tailwind 2
- **Styling**: Tailwind CSS 3
- **Icons**: Heroicons
- **State Management**: React Hooks (useState, useEffect)
- **Design System**: Material Design

## Material Tailwind 컴포넌트 사용

### 🎯 적용된 컴포넌트들

- **Navigation**: `Navbar`, `Menu`, `MenuHandler`, `MenuList`, `MenuItem`
- **Layout**: `Card`, `CardBody`, `CardHeader`
- **Forms**: `Input`, `Button`, `IconButton`
- **Data Display**: `Typography`, `Chip`, `Avatar`
- **Feedback**: `Carousel`, `Progress`
- **Utils**: `Collapse`, `ThemeProvider`

### 🎨 Material Design 특징

- **Elevation 시스템**: 카드와 컴포넌트의 그림자 효과
- **Color System**: Material Design 색상 팔레트
- **Typography Scale**: 일관된 타이포그래피 시스템
- **Component States**: Hover, Focus, Active 상태 표현
- **Smooth Animations**: 부드러운 전환 효과

## 레이아웃 구성

### 1. Header (상단)
- **Material Navbar** 컴포넌트 사용
- 유틸리티 바 (회원가입, 로그인, 원격지원 등)
- 로고 및 **Material Input** 검색 바
- **Material Menu** 드롭다운 네비게이션

### 2. Main Section (중간)
- **Material Carousel** 자동 슬라이드 메인 배너 (3개 슬라이드)
- **Material Card** 퀵메뉴 그리드 (6개 항목)
- **Material Card** 공지사항 섹션 (**Chip** NEW 배지 포함)
- 동물등록제 안내 카드
- 국가봉사동물 입양정보
- 보호동물 현황 카드들
- 지역별 동물병원·보호센터 통계

### 3. Footer (하단)
- 동물사랑배움터 배너
- 동물보호 상담센터 연락처 (**Heroicons** 아이콘 포함)
- 관련기관 **Material Card** 슬라이드 배너
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

## Material Tailwind 테마 설정

Material Tailwind는 다음과 같은 색상 시스템을 제공합니다:

### 기본 색상
- `blue-gray` - 중성 색상
- `blue` - 기본 색상
- `green` - 성공/긍정 색상
- `red` - 경고/오류 색상

### 컴포넌트 변형
- `filled` - 채워진 스타일
- `outlined` - 테두리 스타일
- `text` - 텍스트 스타일

## 컴포넌트 구조

```
src/
├── components/
│   ├── Header.jsx      # Material Navbar 및 네비게이션
│   ├── MainSection.jsx # Material Cards 및 Carousel
│   └── Footer.jsx      # Material Cards 슬라이드 및 연락처
├── App.jsx             # ThemeProvider 적용 메인 애플리케이션
├── main.jsx           # 엔트리 포인트
└── index.css          # Tailwind CSS
```

## 반응형 브레이크포인트

Material Tailwind는 다음 브레이크포인트를 사용합니다:

- **sm**: 640px
- **md**: 768px
- **lg**: 1024px
- **xl**: 1280px
- **2xl**: 1536px

## 커스터마이징

### Material Tailwind 테마 변경
```javascript
// tailwind.config.js에서 withMT() 함수 내에서 커스터마이징
export default withMT({
  theme: {
    extend: {
      colors: {
        // 커스텀 색상 추가
      }
    }
  }
});
```

### 컴포넌트 스타일 수정
각 컴포넌트에서 Material Tailwind의 `className` prop을 사용하여 스타일을 커스터마이징할 수 있습니다.

### 슬라이드 설정
- 메인 배너: `MainSection.jsx`의 `banners` 배열
- 관련기관: `Footer.jsx`의 `partnerSites` 배열

## Material Design vs daisyUI 비교

| 특징 | Material Tailwind | daisyUI |
|------|------------------|---------|
| 디자인 시스템 | Material Design | 자체 디자인 시스템 |
| 컴포넌트 수 | 50+ | 60+ |
| 애니메이션 | Material Motion | CSS 기반 |
| 접근성 | Material 표준 | WAI-ARIA 준수 |
| 번들 크기 | 중간 | 가벼움 |
| 커스터마이징 | 제한적 | 높은 자유도 |

## 라이센스

이 프로젝트는 학습 및 참고 목적으로 제작되었습니다. 