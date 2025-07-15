import React, { useState, useRef, useEffect, forwardRef } from "react";
import { Button, Select, Option } from "@material-tailwind/react";
import { useCommonCodes } from '@/contexts/CommonCodeContext';
import { AllCommunityModule, ModuleRegistry } from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import api from '@/api/api';
import CommonCodeRegister from './CommonCodeRegister';
import ConfirmDialog from '@/components/common/ConfirmDialog';

ModuleRegistry.registerModules([AllCommunityModule]);

export default function CommonCode() {
  const { codes, loading: _, error: __, reloadCodes } = useCommonCodes();
  const gridRef = useRef();
  const [allCodes, setAllCodes] = useState([]);
  const [rowData, setRowData] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedRow, setSelectedRow] = useState(null);
  const [isRegisterOpen, setIsRegisterOpen] = useState(false);
  const [searchCondition, setSearchCondition] = useState({ groupCd: '', useYn: ''});
  const [confirmDialogOption, setConfirmDialogOption] = useState({isOpen: false, title: '', message: '', onConfirm: () => {}});

  const colDefs = [
    { headerName: '그룹코드', field: 'groupCd', width: 120 },
    { headerName: '코드', field: 'cd', width: 80 },
    { headerName: '코드명', field: 'cdNm'},
    { headerName: '코드명(영문)', field: 'cdNmEn'},
    { headerName: 'value1', field: 'value1', width: 200 },
    { headerName: '사용여부', field: 'useYn'},
    { headerName: '코멘트', field: 'comment'}
  ];

  const defaultColDef = {
    flex: 1,
    sortable: true,
    minWidth: 30,
    resizable: true,
    autoSize: true,
    //filter: true,
    //filterParams: {
    //  filterOptions: ['키워드 포함', '키워드와 일치']
    //}
  };

  // 데이터 새로고침
  const handleRefresh = async (globalRefresh = false) => {
    setIsLoading(true);
    try {
      if (globalRefresh) {
        await reloadCodes();
      }
      await loadData();
    } finally {
      setIsLoading(false);
    }
  };

  const loadData = async () => {
    const res = await api.get('/api/common/codes');
    setAllCodes(res.data);
  }

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    setRowData((allCodes || []).filter(item => 
      (searchCondition.groupCd === '' || item.groupCd === searchCondition.groupCd)  
      && (searchCondition.useYn === '' || item.useYn === searchCondition.useYn)
    ));
  }, [allCodes, searchCondition.groupCd, searchCondition.useYn]);

  const handleConfirmRegister = () => {
    setIsRegisterOpen(false);
    handleRefresh();
  };

  // 행 스타일 설정 - useYn이 'N'인 경우 스타일 변경
  const getRowStyle = (params) => {
    if (params.data.useYn === 'N') {
      return { backgroundColor: '#d1d5db' }; // bg-gray-300 색상
    }
    return null;
  };

  const handleDelete = async () => {
    if (!selectedRow) return;
    setConfirmDialogOption({
      isOpen: true,
      title: '공통코드 삭제',
      message: `${selectedRow.groupCd} - ${selectedRow.cdNm} 코드를 삭제하시겠습니까?`,
      onConfirm: handleConfirmDelete
    });
  };

  const handleConfirmDelete = async () => {
    setConfirmDialogOption({isOpen: false, title: '', message: '', onConfirm: () => {}});
    setIsLoading(true);
    try {
      await api.delete('/api/common/code', {
        data: {
          groupCd: selectedRow.groupCd,
          cd: selectedRow.cd
        }
      });
      
      setSelectedRow(null);
      if (gridRef.current?.api) {
        gridRef.current.api.deselectAll();
      }
      handleRefresh();
      
    } catch (error) {
      console.error('공통코드 삭제 실패:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6 w-[90%] mx-auto min-w-[1200px]">
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-2xl font-bold">공통 코드 관리</h1>
      </div>
      <div className="mb-4 flex items-center">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="font-medium">그룹코드:</label>
            <select 
              className="border border-gray-300 rounded px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={searchCondition.groupCd}
              onChange={(e) => {
                setSearchCondition({ ...searchCondition, groupCd: e.target.value });
              }}
            >
              <option value="">전체</option>
              {[...new Set((allCodes || []).map(item => item.groupCd))].sort().map(groupCd => (
                <option key={groupCd} value={groupCd}>{groupCd}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="font-medium">사용여부:</label>
            <select
              className="border border-gray-300 rounded px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={searchCondition.useYn}
              onChange={(e) => {
                setSearchCondition({ ...searchCondition, useYn: e.target.value });
              }}
            >
              <option value="">전체</option>
              <option value="Y">Y</option>
              <option value="N">N</option>
            </select>
          </div>
        </div>
        <div className="ml-auto flex justify-end gap-2 flex-wrap">
          <Button 
            onClick={handleRefresh}
            disabled={isLoading}
            className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 disabled:bg-gray-300 disabled:cursor-not-allowed">
            {isLoading ? '로딩중...' : '새로고침'}
          </Button>
          <Button
             onClick={() => {
               setSelectedRow(null);
               if (gridRef.current?.api) {
                 gridRef.current.api.deselectAll();
               }
               setIsRegisterOpen(true);
             }}
             disabled={isLoading}
             className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-200 disabled:bg-gray-300 disabled:cursor-not-allowed"
           >
             등록
           </Button>
          <Button 
             onClick={() => setIsRegisterOpen(true)}
             disabled={!selectedRow || isLoading}
             className="bg-gray-400 text-white px-6 py-2 rounded hover:bg-gray-700 hover:text-white"
             >
             수정
           </Button>
          <Button
             onClick={handleDelete}
             disabled={!selectedRow || isLoading}
             className="bg-red-400 text-white px-6 py-2 rounded hover:bg-red-600"
           >
             {isLoading ? '삭제 중...' : '삭제'}
           </Button>
        </div>
      </div>
      <div className="h-[70vh] bg-gray-100 rounded-lg">
        <AgGridReact
          ref={gridRef}
          rowData={rowData}
          columnDefs={colDefs}
          defaultColDef={defaultColDef}
          getRowId={(params) => ''.concat(params.data.groupCd, '_', params.data.cd) }
          getRowStyle={getRowStyle}
          rowSelection="single"
          suppressCellSelection={true}
          pagination={true}
          paginationAutoPageSize={true}
          suppressHorizontalScroll={true}
          suppressRowTransform={true}
          onRowDoubleClicked={(params) => {
            setSelectedRow(params.data);
            setIsRegisterOpen(true);
          }}
          onRowClicked={(params) => setSelectedRow(params.data)}
        />
      </div>
      {isRegisterOpen && (
        <CommonCodeRegister
          isOpen={isRegisterOpen}
          onClose={() => setIsRegisterOpen(false)}
          onSaved={handleConfirmRegister}
          selectedRow={selectedRow}
        />
      )}

      {confirmDialogOption.isOpen && (
        <ConfirmDialog
          isOpen={confirmDialogOption.isOpen}
          onClose={() => setConfirmDialogOption({isOpen: false, title: '', message: '', onConfirm: () => {}})}
          onConfirm={confirmDialogOption.onConfirm}
          title={confirmDialogOption.title}
          message={confirmDialogOption.message}
        />
      )}
    </div>
  );
}