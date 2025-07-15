import React, { useState } from 'react';
import SearchMyDog from './SearchMyDog';
import SearchedDogList from './SearchedDogList';
import DogDetailView from './DogDetailView';

function DogSimilaritySearch() {
  const [currentPage, setCurrentPage] = useState('search'); // 'search', 'gallery', 'detail'
  const [searchResults, setSearchResults] = useState(null);
  const [selectedDog, setSelectedDog] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [queryKeypointImage, setQueryKeypointImage] = useState(null);
  const [searchMetadata, setSearchMetadata] = useState(null);

  // SearchMyDog에서 검색 완료 시 호출
  const handleSearchResults = (results, originalImg, queryKeypointImg, metadata) => {
    setSearchResults(results);
    setOriginalImage(originalImg);
    setQueryKeypointImage(queryKeypointImg);
    setSearchMetadata(metadata);
    setCurrentPage('gallery');
  };

  // SearchPetPage에서 강아지 선택 시 호출
  const handleSelectDog = (dog) => {
    setSelectedDog(dog);
    setCurrentPage('detail');
  };

  // 새로운 검색으로 돌아가기
  const handleBackToSearch = () => {
    setCurrentPage('search');
    setSearchResults(null);
    setSelectedDog(null);
    setOriginalImage(null);
    setQueryKeypointImage(null);
    setSearchMetadata(null);
  };

  // 갤러리로 돌아가기 (상세에서)
  const handleBackToGallery = () => {
    setCurrentPage('gallery');
    setSelectedDog(null);
  };

  return (
    <div style={{ minHeight: '100vh' }}>
      {currentPage === 'search' && (
        <SearchMyDog
          onClose={handleBackToSearch}
          onSearchResults={handleSearchResults}
        />
      )}

      {currentPage === 'gallery' && (
        <SearchedDogList
          searchResults={searchResults}
          onSelectDog={handleSelectDog}
          onBackToSearch={handleBackToSearch}
          originalImage={originalImage}
          queryKeypointImage={queryKeypointImage}
        />
      )}

      {currentPage === 'detail' && (
        <DogDetailView
          dogData={selectedDog}
          onBack={handleBackToGallery}
          queryKeypointImage={queryKeypointImage}
          searchMetadata={searchMetadata}
        />
      )}
    </div>
  );
}

export default DogSimilaritySearch;
