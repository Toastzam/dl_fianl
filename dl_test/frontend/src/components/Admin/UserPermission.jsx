import React, { useState, useRef, useEffect, forwardRef } from "react";
import { Button, Select, Option } from "@material-tailwind/react";
import { AllCommunityModule, ModuleRegistry } from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import api from '@/api/api';
import ConfirmDialog from "@/components/common/ConfirmDialog";

ModuleRegistry.registerModules([AllCommunityModule]);

const RoleSelectRenderer = React.memo((props) => {
  const { value, data, onRoleChange } = props;
  
  const handleChange = (e) => {
    const newValue = e.target.value;
    
    if (onRoleChange && data?.userId) {
      onRoleChange(data.userId, newValue);
    }
  };

  return (
    <div className="w-full h-full flex items-center justify-center p-1">
      <select 
        value={value || 'USER'} 
        onChange={handleChange}
        className="w-full h-full border border-gray-300 rounded px-2 text-sm bg-white hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
        style={{ 
          minHeight: '30px',
          fontSize: '13px',
          appearance: 'menulist'
        }}
      >
        <option value="ADMIN">ADMIN</option>
        <option value="USER">USER</option>
      </select>
    </div>
  );
}, (prevProps, nextProps) => {
  // props 비교 함수 - 실제로 값이 변경되었을 때만 재렌더링
  return prevProps.value === nextProps.value && 
         prevProps.data?.userId === nextProps.data?.userId;
});

export default function UserPermission() {
  const gridRef = useRef();
  const [rowData, setRowData] = useState([]);
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const [pendingChange, setPendingChange] = useState(null); // 변경 대기 중인 사용자 정보
  const [isLoading, setIsLoading] = useState(false);

  // 권한 변경 핸들러 - 확인 다이얼로그 표시
  const handleRoleChange = React.useCallback((userId, newRole) => {
    const currentUser = rowData.find(user => user.userId === userId);
    if (currentUser && currentUser.role !== newRole) {
      setPendingChange({
        userId,
        currentRole: currentUser.role,
        newRole,
        userName: currentUser.name || currentUser.userId
      });
      setIsConfirmOpen(true);
    }
  }, [rowData]);

  // 확인 다이얼로그에서 '확인' 클릭 시 실행
  const handleConfirm = async () => {
    if (!pendingChange) return;

    try {
      // API 호출하여 권한 즉시 변경
      await api.post('/api/user/update-role', {
        userId: pendingChange.userId,
        role: pendingChange.newRole
      });

      // 성공 시 로컬 데이터 업데이트
      setRowData(prevData => 
        prevData.map(user => 
          user.userId === pendingChange.userId 
            ? { ...user, role: pendingChange.newRole }
            : user
        )
      );

      alert(`${pendingChange.userName}의 권한이 ${pendingChange.currentRole} → ${pendingChange.newRole}로 변경되었습니다.`);
      
    } catch (error) {
      console.error('권한 변경 실패:', error);
      alert('권한 변경 중 오류가 발생했습니다.');
      
      // 실패 시 그리드 데이터 다시 로드하여 원상복구
      await loadData();
    } finally {
      setIsConfirmOpen(false);
      setPendingChange(null);
    }
  };

  // cellRendererParams를 별도로 정의하여 재렌더링 방지
  const roleRendererParams = React.useMemo(() => ({
    onRoleChange: handleRoleChange
  }), [handleRoleChange]);

  // 컬럼 정의를 useMemo로 최적화
  const colDefs = React.useMemo(() => [
    { headerName: '사용자ID', field: 'userId', width: 120 },
    { headerName: '이름', field: 'name', width: 100 },
    { headerName: '이메일', field: 'email', flex: 1 },
    { headerName: '연락처', field: 'phone', width: 130 },
    { 
      headerName: '권한', 
      field: 'role', 
      cellRenderer: RoleSelectRenderer,
      cellRendererParams: roleRendererParams,
      sortable: false,
      filter: false,
      cellStyle: { 
        textAlign: 'center',
        padding: '2px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      },
      width: 120
    },
    { 
      headerName: 'SSO연결', 
      field: 'provider', 
      valueFormatter: (params) => {
        return params.value === 'LOCAL' ? '' : params.value;
      },
      width: 100,
      cellStyle: { textAlign: 'center' }
    },
    { headerName: '가입일자', field: 'createdAt', valueFormatter: (params) => {
        return params.value ? new Date(params.value).toLocaleDateString() : '';
      } 
    }
  ], [roleRendererParams]);

  const defaultColDef = {
    flex: 1,
    sortable: true,
    minWidth: 30,
    resizable: true,
    autoSize: true,
  };

  const gridOptions = {
    rowHeight: 45, // 행 높이 증가
    headerHeight: 40, // 헤더 높이
  };

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setIsLoading(true);
    setRowData([]);
    try {
      const res = await api.get('/api/user/list');
      setRowData(res?.data || []);
    } catch (e) {
      console.error('사용자 목록 로드 실패:', e);
      alert('사용자 목록을 불러오는데 실패했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  // 데이터 새로고침
  const handleRefresh = async () => {
    await loadData();
  };


  return (
    <div className="p-6 w-[80%] mx-auto min-w-[1000px]">
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-2xl font-bold">사용자 권한 관리</h1>
        <div className="space-x-2">
          <Button 
            onClick={handleRefresh}
            disabled={isLoading}
            className="bg-green-500 text-white px-6 py-2 rounded hover:bg-green-600 disabled:bg-gray-300 disabled:cursor-not-allowed">
            {isLoading ? '로딩중...' : '새로고침'}
          </Button>
        </div>
      </div>
      <div className="h-[70vh] bg-gray-100 rounded-lg">
        <AgGridReact
          ref={gridRef}
          rowData={rowData}
          columnDefs={colDefs}
          defaultColDef={defaultColDef}
          rowHeight={gridOptions.rowHeight}
          headerHeight={gridOptions.headerHeight}
          getRowId={(params) => params.data.userId} // 각 행을 고유하게 식별
          rowSelection="single"
          suppressCellSelection={true}
          pagination={true}
          paginationAutoPageSize={true}
          suppressHorizontalScroll={true}
          animateRows={true} // 행 애니메이션 활성화
          suppressRowTransform={true} // row 변환 억제
        />
      </div>

      {isConfirmOpen && pendingChange && (
        <ConfirmDialog
          isOpen={isConfirmOpen}
          onClose={() => {
            setIsConfirmOpen(false);
            setPendingChange(null);
          }}
          onConfirm={handleConfirm}
          title="사용자 권한 변경 확인"
          message={`${pendingChange.userName}님의 권한을 "${pendingChange.currentRole}"에서 "${pendingChange.newRole}"로 변경하시겠습니까?`}
        />
      )}
    </div>
  );
}
