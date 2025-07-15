import React, { useState, useEffect } from 'react';
import {
  Card,
  CardBody,
  Typography,
  Button,
  Chip,
  IconButton,
} from "@material-tailwind/react";
import { 
  ArrowLeftIcon, 
  ArrowRightIcon,
  PhoneIcon,
  MapPinIcon,
  ClockIcon,
} from "@heroicons/react/24/outline";

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
    phone: '031-239-5855',
    systemInquiry: '054-810-8626',
    address: '경기도 수원시 팔달구 인계동 번지 신관 3층 208-5 KR 풍림빌딩',
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

  const totalSlides = Math.ceil(partnerSites.length / 3);

  const nextSlide = () => {
    setCurrentPartnerSlide((prev) => (prev + 1) % totalSlides);
  };

  const prevSlide = () => {
    setCurrentPartnerSlide((prev) => (prev - 1 + totalSlides) % totalSlides);
  };

  return (
    <footer className="bg-blue-gray-50 mt-16">

      {/* 하단 정보 */}
      <section className="bg-blue-gray-100 py-8 border-t border-blue-gray-200">
        <div className="container mx-auto px-4">
          <div className="flex flex-col lg:flex-row justify-between items-start gap-8">
            {/* 링크 메뉴 */}
            <div className="mb-6 lg:mb-0">
              <nav className="flex flex-wrap gap-6 mb-4">
                {footerLinks.map((link, index) => (
                  <Typography
                    key={index}
                    as="a"
                    href={link.url}
                    variant="small"
                    className="text-blue-gray-600 hover:text-blue-500 transition-colors cursor-pointer"
                  >
                    {link.name}
                  </Typography>
                ))}
              </nav>
            </div>

            {/* 연락처 정보 */}
            <div className="text-blue-gray-600">
              <Typography variant="h6" className="font-bold mb-3 text-blue-gray-800">
                휴먼IT교육센터 2팀
              </Typography>
              <div className="space-y-2">
                <div className="flex items-start gap-2">
                  <MapPinIcon className="h-4 w-4 mt-0.5 flex-shrink-0" />
                  <Typography variant="small">
                    {contactInfo.address}
                  </Typography>
                </div>
                <div className="flex items-center gap-2">
                  <PhoneIcon className="h-4 w-4" />
                  <Typography variant="small">
                    전화번호: {contactInfo.phone} 
                  </Typography>
                </div>
                <Typography variant="small" className="text-blue-gray-500 mt-4">
                  Copyright by Animal and Plant Quarantine Agency. All Rights Reserved.
                </Typography>
              </div>
            </div>
          </div>
        </div>
      </section>
    </footer>
  );
};

export default Footer; 