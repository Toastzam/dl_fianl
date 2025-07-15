// ColorPreview.jsx
import React from "react";
import { Button } from "@material-tailwind/react";
import { Typography } from "@material-tailwind/react";
// Tailwind 설정 로드
// import tailwindConfig from "@/../tailwind.config.js";
import tailwindConfig from "@tailwind-config";


// 기본 색상명 유지 (group: basic)
const BASIC_COLOR_NAMES = [
  "white",
  "black",
  "blue-gray",
  "gray",
  "brown",
  "deep-orange",
  "orange",
  "amber",
  "yellow",
  "lime",
  "light-green",
  "green",
  "teal",
  "cyan",
  "light-blue",
  "blue",
  "indigo",
  "deep-purple",
  "purple",
  "pink",
  "red",
];

// 그룹당 최대 2줄(16칸)만 표시
const MAX_PREVIEW_CELLS = 16;

export default function ColorPreview() {
  const themeColors = Object.fromEntries(
    Object.entries(tailwindConfig.theme?.colors ?? {})
      .filter(([key]) => !['inherit', 'current'].includes(key))
      .filter(([_, value]) => typeof value !== 'string')
  );
  console.log(themeColors);

  // basic 그룹 데이터 준비
  // const basicColors = BASIC_COLOR_NAMES.map((name) => {
  //   const value = themeColors[name];
  //   // 객체라면 대표로 500번대 색상 사용, 없으면 그대로 Tailwind class 사용
  //   if (typeof value === "object") {
  //     return { label: name, hex: value["500"] ?? Object.values(value)[0] };
  //   }
  //   if (typeof value === "string") {
  //     return { label: name, hex: value };
  //   }
  //   // theme.colors 에 존재하지 않는 경우 – Tailwind 기본 클래스 사용하도록 null 반환
  //   return { label: name, hex: null };
  // });

  // 나머지 색상 그룹 (basic 제외)
  const otherGroups = Object.entries(themeColors).map(([group, shades]) => ({
    group,
    shades: Object.entries(shades).slice(0, MAX_PREVIEW_CELLS), // 최대 16개 제한
  }));
  
  // 선택한 테마 컬러
  const THEME_COLOR_GROUPS = ['blue','amber','gray'];

  return (
    <div class="container" className="p-8">
      <Typography variant="small" className="mb-4 font-semibold text-gray-700">
        ※ 사용가능한 색상 목록입니다. <br />
        사용예: <Button className="bg-blue-300 text-white border-gray-300">blue-500</Button>&nbsp;
      <code className="bg-gray-100 px-2 py-1 rounded text-sm font-mono border border-gray-300">
        {'<Button className="bg-blue-300 text-white border-gray-300">blue-500</Button>'}
      </code>
      </Typography>
      <div class="theme-color" className="h-[50%] overflow-auto p-4 space-y-8 border border-gray-300">
        {otherGroups.filter(({ group }) => THEME_COLOR_GROUPS.includes(group)).map(({ group, shades }) => (
          <section key={group}>
            <Typography variant="small" className="mb-4 font-semibold text-gray-700">
              {group}
            </Typography>
            <div className="grid grid-cols-4 sm:grid-cols-6 lg:grid-cols-8 gap-4">
              {shades.map(([shadeKey, hex]) => (
                <div key={`${group}-${shadeKey}`} className="text-center">
                  <div
                    className="w-16 h-16 rounded-lg shadow-md border border-gray-200"
                    style={{ backgroundColor: hex }}
                  />
                  <Typography variant="small" className="mt-2 font-normal truncate">
                    {`${group}-${shadeKey}`}
                  </Typography>
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>
      <span>&nbsp;</span>
      <div class="all-color"className="h-[40%] overflow-auto p-4 space-y-8 border border-gray-200">
        {/* basic 그룹 */}
        <section>
          <Typography variant="small" className="mb-4 font-semibold text-gray-700">
            basic
          </Typography>
          <div className="grid grid-cols-4 sm:grid-cols-6 lg:grid-cols-8 gap-4">
            {BASIC_COLOR_NAMES.map(color => (
            <div key={color} className="text-center">
              <div
                className={`w-16 h-16 rounded border border-gray-200 
                  ${color === 'white' ? 'bg-white' 
                    : color === 'black' ? 'bg-black' 
                    : `bg-${color}-500`}`}
              />
              {/* <span className="block mt-1 text-xs truncate">{color}</span> */}
              <Typography variant="small" className="mt-2 font-normal truncate">
                  {color}
                </Typography>
            </div>
          ))}
          </div>
        </section>

        {/* 기타 그룹 */}
        {otherGroups.map(({ group, shades }) => (
          <section key={group}>
            <Typography variant="small" className="mb-4 font-semibold text-gray-700">
              {group}
            </Typography>
            <div className="grid grid-cols-4 sm:grid-cols-6 lg:grid-cols-8 gap-4">
              {shades.map(([shadeKey, hex]) => (
                <div key={`${group}-${shadeKey}`} className="text-center">
                  <div
                    className="w-16 h-16 rounded-lg shadow-md border border-gray-200"
                    style={{ backgroundColor: hex }}
                  />
                  <Typography variant="small" className="mt-2 font-normal truncate">
                    {`${group}-${shadeKey}`}
                  </Typography>
                </div>
              ))}
            </div>
          </section>
        ))}
        <div className="h-16"></div>
      </div>
    </div>
  );
}
