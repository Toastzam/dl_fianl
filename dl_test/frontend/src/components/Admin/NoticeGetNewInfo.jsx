import { useState, useEffect } from "react";
import api from "@/api/api"; // axios 인스턴스 사용

export default function RecentRescueDogsPage() {
  const [petProfiles, setPetProfiles] = useState([]);
  const [petFindInfos, setPetFindInfos] = useState([]);
  const [selected, setSelected] = useState(new Set());

  useEffect(() => {
    async function fetchData() {
      try {
        const resProfiles = await api.get("/admin/notice/petprofile");
        setPetProfiles(resProfiles.data);

        const resFindInfos = await api.get("/admin/notice/petfindinfo");
        setPetFindInfos(resFindInfos.data);
      } catch (error) {
        console.error("데이터 로딩 실패:", error);
      }
    }

    fetchData();
  }, []);

  function toggleSelect(key) {
    setSelected((prev) => {
      const newSet = new Set(prev);
      newSet.has(key) ? newSet.delete(key) : newSet.add(key);
      return newSet;
    });
  }

  return (
    <div className="p-6 space-y-12">
      {/* 최근 등록된 보호견 정보 */}
      <div>
        <h2 className="text-xl font-bold mb-3">최근 등록된 보호견 정보</h2>
        <table className="w-full border border-gray-300 text-sm">
          <thead>
            <tr className="bg-gray-100">
              <th className="py-2 px-2 border">이름</th>
              <th className="py-2 px-2 border">품종</th>
              <th className="py-2 px-2 border">발견장소</th>
              <th className="py-2 px-2 border">보호소 이름</th>
              <th className="py-2 px-2 border">보호소 등록일</th>
              <th className="py-2 px-2 border">이미지 링크</th>
            </tr>
          </thead>
          <tbody>
            {petProfiles.map((dog) => (
              <tr key={`profile-${dog.id}`} className="hover:bg-gray-50">
                <td className="py-2 px-2 border text-center">{dog.name}</td>
                <td className="py-2 px-2 border text-center">{dog.breed}</td>
                <td className="py-2 px-2 border text-center">{dog.foundplace}</td>
                <td className="py-2 px-2 border text-center">{dog.shelter}</td>
                <td className="py-2 px-2 border text-center">{dog.receptionDate}</td>
                <td className="py-2 px-2 border text-center">
                  {dog.publicUrl ? (
                    <a
                      href={dog.publicUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 underline"
                    >
                      이미지 보기
                    </a>
                  ) : (
                    "없음"
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 최근 발견 신고된 보호견 정보 */}
      <div>
        <h2 className="text-xl font-bold mb-3">최근 발견 신고된 보호견 정보</h2>
        <table className="w-full border border-gray-300 text-sm">
          <thead>
            <tr className="bg-gray-100">
              <th className="py-2 px-2 border">품종</th>
              <th className="py-2 px-2 border">발견장소</th>
              <th className="py-2 px-2 border">발견시간</th>
              <th className="py-2 px-2 border">신고 등록된 보호소 이름</th>
              <th className="py-2 px-2 border">보호소 등록일</th>
            </tr>
          </thead>
          <tbody>
            {petFindInfos.map((dog) => (
              <tr key={`find-${dog.id}`} className="hover:bg-gray-50">
                <td className="py-2 px-2 border text-center">{dog.breed}</td>
                <td className="py-2 px-2 border text-center">{dog.foundplace}</td>
                <td className="py-2 px-2 border text-center">{dog.foundtime}</td>
                <td className="py-2 px-2 border text-center">{dog.shelter}</td>
                <td className="py-2 px-2 border text-center">{dog.receptionDate}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
