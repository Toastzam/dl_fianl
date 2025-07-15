import React, { useState, useRef, useEffect, forwardRef } from "react";
import { Button, Tooltip } from "@material-tailwind/react";
import { AllCommunityModule, ModuleRegistry } from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
// import 'ag-grid-community/styles/ag-grid.css';
// import 'ag-grid-community/styles/ag-theme-material.css'; //alpine, balham,material, quartz 
import PetRegister from './PetRegister';
import AlertDialog from '../common/AlertDialog';
import ConfirmDialog from '../common/ConfirmDialog';
import api from '@/api/api';

ModuleRegistry.registerModules([AllCommunityModule]);

// 프로필 팝업 표시
const _popupProfile = (html) => {
  try {
    // Blob을 사용해서 URL 생성
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    
    // 새 탭에서 열기
    const newWindow = window.open(url, '_blank');
    
    if (newWindow) {
      // 메모리 정리 (5초 후)
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    } else {
      setAlertOption({isOpen: true, message: '팝업을 허용해야 보실 수 있습니다.', title: '알림'});
      // 팝업 차단 시 다운로드로 대체
      // const link = document.createElement('a');
      // link.href = url;
      // link.download = 'result.html';
      // link.click();
      // URL.revokeObjectURL(url);
    }
  } catch (error) {
    console.error('HTML 표시 실패:', error);
    console.log('HTML 내용:', html);
  }
};

// Profile Button 컴포넌트 정의
const ProfileButtonRenderer = forwardRef((props, ref) => {
  const { value } = props;
  
  const handleClick = () => {
    if (value) {
      _popupProfile(value);
    }
  };

  return value ? (
    <Tooltip content="프로필 팝업 보기" placement="bottom">
      <Button
        size="sm"
        variant="text"
        className="bg-blue-700 hover:bg-blue-300 text-white"
        disabled={!value}
        onClick={handleClick}
      >
        보기
      </Button>
    </Tooltip>
  ) : null;
});


export default function PetManagement() {
  const gridRef = useRef();
  const [isRegisterOpen, setIsRegisterOpen] = useState(false);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [alertOption, setAlertOption] = useState({isOpen: false, message: '', title: ''});
  const [selectedPet, setSelectedPet] = useState(null);
  const [rowData, setRowData] = useState([]);

  const [colDefs, setColDefs] = useState([
    { headerName: 'PetID', field: 'noticeId', minWidth: 185, renderCell: (params) => {
      return `${params.data.noticeId || params.data.petUid}`;
    } },
    { headerName: 'ID', field: 'petUid', floatingFilter: false, hide: true },
    { headerName: '이름', field: 'name'},
    { headerName: '성별', field: 'genderNm' },
    { headerName: '견종', field: 'breedNm' },
    { headerName: '나이', field: 'birthYyyyMm' },
    // { headerName: '중성화여부', field: 'neuteredNm' },
    // { headerName: '무게(kg)', field: 'weightKg' },
    // { headerName: '털색', field: 'color', tooltipField: 'color' },
    { headerName: '입양상태', field: 'adoptionStatusNm' },
    { headerName: '사진 수', field: 'imageCount'},
    { 
      headerName: '프로필', 
      field: 'profileHtml', 
      cellRenderer: ProfileButtonRenderer,
      cellRendererParams: {
        suppressCount: true
      },
      sortable: false,
      filter: false,
      cellStyle: { textAlign: 'center' },
    },
    { headerName: '보호센터', field: 'shelterName', tooltipField: 'shelterName' },
    { headerName: '관할기관', field: 'jurisdictionOrg', tooltipField: 'jurisdictionOrg' },
    { headerName: '접수일자', field: 'receptionDate', minWidth: 100 },
  ]);

  const defaultColDef = {
    flex: 1,
    sortable: true,
    minWidth: 30,
    resizable: true,
    tooltipValueGetter: (params) => {
      if (params.colDef.field === 'profileHtml') {
        return null;
      }
      // 텍스트가 길 때만 툴팁 표시
      const text = params.value || '';
      return text.length > 20 ? text : null;
    },
    autoSize: true,
  };

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setRowData([]);
    api.get('/manager/pet-list')
    .then(res => {
      setSelectedPet(null);
      setRowData(res?.data);
    })
    .catch(e => {
      console.error(e);
    });
  }

  const _popupProfile = (html) => {
    console.log('html => ', html);
    try {
      // Blob을 사용해서 URL 생성
      const blob = new Blob([html], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      
      // 새 탭에서 열기
      const newWindow = window.open(url, '_blank');
      
      if (newWindow) {
        // 메모리 정리 (5초 후)
        setTimeout(() => URL.revokeObjectURL(url), 5000);
      } else {
        // 팝업 차단 시 다운로드로 대체
        const link = document.createElement('a');
        link.href = url;
        link.download = 'result.html';
        link.click();
        URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('HTML 표시 실패:', error);
      console.log('HTML 내용:', html);
    }
  }

  const handleConfirmRegister = () => {
    setIsRegisterOpen(false);
    // setAlertOption({isOpen: true, message: `${selectedPet?.pedUid ? '수정' : '등록'}되었습니다.`, title: '저장 완료'});
    loadData();
  };

  const handleDeleteClick = () => {
    const selectedNodes = gridRef.current.api.getSelectedNodes();
    if (selectedNodes.length === 0) {
      setAlertOption({isOpen: true, message: '삭제할 항목을 선택해주세요.', title: '알림'});
      return;
    }
    const pet = selectedNodes[0].data;
    setSelectedPet(pet);
    setIsDeleteDialogOpen(true);
  };

  const handleConfirmDelete = async () => {
    try {
      await api.delete(`/manager/pet/${selectedPet.petUid}`)
      .then(res => {
        setAlertOption({isOpen: true, message: `${selectedPet.noticeId}이(가) 삭제되었습니다.`, title: '삭제 완료'});
        loadData();
        setSelectedPet(null);
        setIsDeleteDialogOpen(false);
      })
      .catch(e => {
        console.error(e);
      });
    } catch (error) {
      console.error('데이터 삭제 실패:', error);
      setAlertOption({
        isOpen: true,
        message: '${selectedPet.noticeId} 삭제에 실패했습니다.',
        title: '삭제 실패'
      });
    }
  };

  const handleRefresh = async () => {
    await loadData();
  };

  return (
    <div className="p-6">      
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-2xl font-bold">보호견 정보 관리</h1>
        <div className="space-x-2">
          <Button 
            onClick={handleRefresh}
            className="bg-green-500 text-white px-6 py-2 rounded hover:bg-green-600 disabled:bg-gray-300 disabled:cursor-not-allowed">
            새로고침
          </Button>
          <Button
            onClick={() => {
              setSelectedPet(null);
              gridRef.current.api.deselectAll();
              setIsRegisterOpen(true);
            }}
            className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-200"
          >
            등록
          </Button>
          <Button 
            onClick={() => setIsRegisterOpen(true)}
            className="bg-gray-400 text-white px-6 py-2 rounded hover:bg-gray-700 hover:text-white"
            disabled={!selectedPet}
            >
            수정
          </Button>
          <Button
            onClick={handleDeleteClick}
            className="bg-red-400 text-white px-6 py-2 rounded hover:bg-red-600"
          >
            삭제
          </Button>
        </div>
      </div>
      <div className="h-[70vh] bg-gray-100 rounded-lg w-[100%]">
        <AgGridReact
          ref={gridRef}
          rowData={rowData}
          columnDefs={colDefs}
          defaultColDef={defaultColDef}
          rowSelection="single"
          cellSelection={false}
          suppressCellFocus={true}
          pagination={true}
          paginationAutoPageSize={true}
          onRowDoubleClicked={(params) => {
            setSelectedPet(params.data);
            setIsRegisterOpen(true);
          }}
          suppressHorizontalScroll={true}
          onRowClicked={(params) => setSelectedPet(params.data)}
        />
      </div>

      {isRegisterOpen && (
        <PetRegister 
          isOpen={isRegisterOpen} 
          onClose={() => setIsRegisterOpen(false)} 
          onSaved={handleConfirmRegister} 
          petUid={selectedPet?.petUid || ''}
        />
      )}

      {isDeleteDialogOpen && (
        <ConfirmDialog
          isOpen={isDeleteDialogOpen}
          onClose={() => setIsDeleteDialogOpen(false)}
          onConfirm={handleConfirmDelete}
          title="보호동물 삭제"
          message={`${selectedPet?.noticeId}를 삭제하시겠습니까?`}
        />
      )}
      
      {alertOption.isOpen && (
        <AlertDialog
          isOpen={alertOption.isOpen}
          onClose={() => setAlertOption({isOpen: false, message: '', title: ''})}
          message={alertOption.message}
          title={alertOption.title}
        />
      )}
    </div>
  );
} 