import React, { useEffect, useState } from "react";
import api from '@/api/api';

const PetList = () => {
  const [recentPets, setRecentPets] = useState([]);
  const [allPets, setAllPets] = useState([]);
  const [filteredPets, setFilteredPets] = useState([]);
  const [breeds, setBreeds] = useState([]);
  const [shelters, setShelters] = useState([]);
  const [selectedBreed, setSelectedBreed] = useState("");
  const [selectedShelter, setSelectedShelter] = useState("");
  const [hasProfile, setHasProfile] = useState(""); // 상세 프로필 여부 상태
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  const formatGender = (code) => {
    switch (code) {
      case "M":
        return "수컷";
      case "F":
        return "암컷";
      case "Q":
      default:
        return "미상";
    }
  };

  useEffect(() => {
    api.post("/petcare/petlist/recent")
      .then((res) => setRecentPets(res.data))
      .catch((err) => console.error("recent pets fetch error", err));

      api.post("/petcare/petlist/random")
      .then((res) => {
        setAllPets(res.data);
        setFilteredPets(res.data);
      })
      .catch((err) => console.error("all pets fetch error", err));

      api.post("/petcare/petlist/breeds")
      .then((res) => setBreeds(res.data))
      .catch((err) => console.error("breeds fetch error", err));

      api.post("/petcare/petlist/shelters")
      .then((res) => setShelters(res.data))
      .catch((err) => console.error("shelters fetch error", err));
  }, []);

  const handleSearch = () => {
    // hasProfile 값 변환: "" => undefined, "yes" => true, "no" => false
    let profileFlag = undefined;
    if (hasProfile === "yes") profileFlag = true;
    else if (hasProfile === "no") profileFlag = false;

    const payload = {
      breed: selectedBreed,
      shelter: selectedShelter,
      hasProfile: profileFlag,
    };

    api.post("/petcare/petlist/search", payload)
      .then((res) => {
        setFilteredPets(res.data);
        setAllPets([]);
        setCurrentPage(1);
      })
      .catch((err) => console.error("검색 요청 에러", err));
  };

  const handleOpenProfile = (htmlContent) => {
    const newWindow = window.open("", "_blank", "width=800,height=600");
    if (newWindow) {
      newWindow.document.open();
      newWindow.document.write(htmlContent);
      newWindow.document.close();
    } else {
      alert("팝업 차단을 해제해주세요.");
    }
  };

  const renderPetCard = (pet, index) => (
    <li
      key={index}
      className="p-4 border rounded shadow flex gap-4 items-center text-left flex-col md:flex-row"
    >
      <img
        src={pet.publicUrl}
        alt={pet.name}
        className="w-24 h-24 object-cover rounded-full"
      />
      <div className="flex-1">
        {pet.name && <p><strong>이름:</strong> {pet.name}</p>}
        <p><strong>품종:</strong> {pet.breedName}</p>
        {pet.birthYyyyMm && <p><strong>출생년월:</strong> {pet.birthYyyyMm}</p>}
        <p><strong>성별:</strong> {formatGender(pet.genderCd)}</p>
        {pet.foundLocation && <p><strong>발견장소:</strong> {pet.foundLocation}</p>}
        <p><strong>보호소:</strong> {pet.shelterName}</p>
        <p><strong>등록일:</strong> {new Date(pet.receptionDate).toLocaleDateString()}</p>

        {pet.profilehtml && (
          <button
            onClick={() => handleOpenProfile(pet.profilehtml)}
            className="mt-3 inline-block bg-amber-900 text-white px-3 py-1 rounded hover:bg-pink-600"
          >
            상세보기
          </button>
        )}
      </div>
    </li>
  );

  const indexOfLastItem = currentPage * itemsPerPage;
  const currentPets = filteredPets.slice(indexOfLastItem - itemsPerPage, indexOfLastItem);
  const totalPages = Math.ceil(filteredPets.length / itemsPerPage);

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      {/* 최근 보호견 */}
      <h2 className="text-xl font-bold mb-4 text-center">최근 등록된 보호견</h2>
      <ul className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {recentPets.map(renderPetCard)}
      </ul>

      {/* 검색 드롭다운 */}
      <div className="mb-6 flex flex-wrap gap-4 justify-center items-center">
        <select
          value={selectedBreed}
          onChange={(e) => setSelectedBreed(e.target.value)}
          className="border p-2 rounded"
        >
          <option value="">품종 선택</option>
          {breeds.map((breed, idx) => (
            <option key={idx} value={breed.breedname}>
              {breed.breedname}
            </option>
          ))}
        </select>

        <select
          value={selectedShelter}
          onChange={(e) => setSelectedShelter(e.target.value)}
          className="border p-2 rounded"
        >
          <option value="">보호소 선택</option>
          {shelters.map((shelter, idx) => (
            <option key={idx} value={shelter.sheltername}>
              {shelter.sheltername}
            </option>
          ))}
        </select>

        <select
          value={hasProfile}
          onChange={(e) => setHasProfile(e.target.value)}
          className="border p-2 rounded"
        >
          <option value="">상세 프로필 여부</option>
          <option value="yes">프로필 있음</option>
          <option value="no">프로필 없음</option>
        </select>

        <button
          onClick={handleSearch}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          검색
        </button>
      </div>

      {/* 보호견 목록 */}
      <h3 className="text-lg font-semibold mb-3 text-center">보호견 목록</h3>
      {filteredPets.length === 0 ? (
        <p className="text-center text-gray-500">검색된 보호견이 없습니다.</p>
      ) : (
        <ul className="space-y-4">
          {currentPets.map(renderPetCard)}
        </ul>
      )}

      {/* 페이지네이션: 검색 결과가 있고 1페이지 이상일 때만 렌더링 */}
      {filteredPets.length > 0 && totalPages > 1 && (
        <div className="mt-6 flex justify-center gap-2">
          {Array.from({ length: totalPages }, (_, i) => (
            <button
              key={i}
              onClick={() => setCurrentPage(i + 1)}
              className={`px-3 py-1 border rounded ${currentPage === i + 1
                ? "bg-blue-500 text-white"
                : "bg-white text-black"
              }`}
            >
            {i + 1}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default PetList;
