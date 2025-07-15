import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogHeader,
  DialogBody,
  DialogFooter,
  Button,
  Input,
  Typography,
  Textarea,
  Select,
  Option,
} from "@material-tailwind/react";
import api from '@/api/api';
import AlertDialog from '@/components/common/AlertDialog';
import { useCommonCodes } from '@/contexts/CommonCodeContext';

const initFormData = {
  groupCd: '',
  cd: '',
  cdNm: '',
  cdNmEn: '',
  value1: '',
  useYn: 'Y',
  comment: '',
}

export default function CommonCodeRegister({ isOpen, onClose, onSaved, selectedRow }) {
  const { codes } = useCommonCodes();
  const [formData, setFormData] = useState(initFormData);
  const [isEditMode, setIsEditMode] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [alertOption, setAlertOption] = useState({isOpen: false, message: '', title: ''});

  useEffect(() => {
    if (isOpen) {
      if (selectedRow) {
        // 수정 모드
        setIsEditMode(true);
        setFormData({
          groupCd: selectedRow.groupCd || '',
          cd: selectedRow.cd || '',
          cdNm: selectedRow.cdNm || '',
          cdNmEn: selectedRow.cdNmEn || '',
          value1: selectedRow.value1 || '',
          useYn: selectedRow.useYn || 'Y',
          comment: selectedRow.comment || '',
        });
      } else {
        // 등록 모드
        setIsEditMode(false);
        setFormData(initFormData);
      }
    }
  }, [isOpen, selectedRow]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const validateForm = () => {
    if (!formData.groupCd.trim()) {
      setAlertOption({
        isOpen: true,
        message: '그룹코드를 선택해주세요.',
        title: '입력 오류'
      });
      return false;
    }
    if (!formData.cd.trim()) {
      setAlertOption({
        isOpen: true,
        message: '코드를 입력해주세요.',
        title: '입력 오류'
      });
      return false;
    }
    if (!formData.cdNm.trim()) {
      setAlertOption({
        isOpen: true,
        message: '코드명을 입력해주세요.',
        title: '입력 오류'
      });
      return false;
    }
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();    
    if (!validateForm()) return;

    setIsLoading(true);
    try {
      const apiUrl = '/api/common/code';
      const method = isEditMode ? 'put' : 'post';
      
      const res = await api[method](apiUrl, formData);
      console.log(">>>> res", res);
      
      if (res?.data?.success || false) {
        setAlertOption({
          isOpen: true,
          message: `공통코드가 성공적으로 ${isEditMode ? '수정' : '등록'}되었습니다.`,
          title: '성공',
        });
        onSaved();
        
      } else {
        setAlertOption({
          isOpen: true,
          message: res?.data?.error || '공통코드 저장 중 오류가 발생했습니다.',
          title: `${isEditMode ? '수정' : '등록'} 오류`,
        });
      }
    } catch (error) {
      console.error('공통코드 저장 실패:', error);
      setAlertOption({
        isOpen: true,
        message: `공통코드 ${isEditMode ? '수정' : '등록'} 중 오류가 발생했습니다.`,
        title: '오류'
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <Dialog open={isOpen} handler={onClose} size="md" className="z-50">
        <DialogHeader className="flex justify-between items-center">
          <span className="text-xl font-semibold">
            {isEditMode ? '공통코드 수정' : '공통코드 등록'}
          </span>
          <Button variant="text" color="blue" size="sm" onClick={onClose} className="p-1">✕</Button>
        </DialogHeader>

        <DialogBody className="max-h-[70vh] overflow-y-auto">
          <form className="space-y-4">
            <div className="grid grid-cols-1 gap-4">
              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">
                  <span className="text-red-500">*</span>그룹코드
                </Typography>
                <Select
                  name="groupCd"
                  value={formData.groupCd}
                  onChange={(value) => handleChange({ target: { name: 'groupCd', value }})}
                  placeholder="그룹코드 선택"
                  disabled={isEditMode}
                >
                  {[...new Set((codes || []).map(item => item.groupCd))]
                    .sort()
                    .map(groupCd => (
                      <Option key={groupCd} value={groupCd}>
                        {groupCd}
                      </Option>
                    ))
                  }
                </Select>
              </div>

              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">
                  <span className="text-red-500">*</span>코드
                </Typography>
                <Input
                  name="cd"
                  value={formData.cd}
                  onChange={handleChange}
                  placeholder="예: M"
                  maxLength={20}
                  disabled={isEditMode} // 수정 모드에서는 코드 변경 불가
                />
              </div>

              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">
                  <span className="text-red-500">*</span>코드명
                </Typography>
                <Input
                  name="cdNm"
                  value={formData.cdNm}
                  onChange={handleChange}
                  placeholder="예: 수컷"
                  maxLength={100}
                />
              </div>

              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">
                  코드명(영문)
                </Typography>
                <Input
                  name="cdNmEn"
                  value={formData.cdNmEn}
                  onChange={handleChange}
                  placeholder="예: Male"
                  maxLength={100}
                />
              </div>

              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">
                  값1
                </Typography>
                <Input
                  name="value1"
                  value={formData.value1}
                  onChange={handleChange}
                  placeholder="추가 값"
                  maxLength={100}
                />
              </div>

              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">
                  사용여부
                </Typography>
                <Select
                  name="useYn"
                  value={formData.useYn || 'Y'}
                  onChange={(value) => handleChange({ target: { name: 'useYn', value }})}
                  className="!border-t-blue-gray-200 focus:!border-t-gray-900"
                >
                  <Option value="Y">사용</Option>
                  <Option value="N">미사용</Option>
                </Select>
              </div>

              <div className="grid grid-cols-[120px_1fr] gap-2 items-center">
                <Typography variant="small" className="text-right font-medium text-blue-gray-900">
                  코멘트
                </Typography>
                <Textarea
                  name="comment"
                  value={formData.comment}
                  onChange={handleChange}
                  placeholder="설명 또는 비고사항"
                  rows={2}
                  maxLength={500}
                />
              </div>
            </div>
          </form>
        </DialogBody>

        <DialogFooter className="flex-shrink-0 border-t border-gray-200 bg-gray-50">
          <div className="flex justify-end space-x-2">
            <Button
              variant="text"
              onClick={onClose}
              disabled={isLoading}
              className="bg-gray-400 hover:bg-gray-600 text-white"
            >
              취소
            </Button>
            <Button
              onClick={handleSubmit}
              disabled={isLoading}
              className="bg-blue-600 hover:bg-blue-700 text-white"
            >
              {isLoading ? '저장 중...' : (isEditMode ? '저장' : '등록')}
            </Button>
          </div>
        </DialogFooter>
      </Dialog>

      <AlertDialog
        isOpen={alertOption.isOpen}
        onClose={() => setAlertOption({isOpen: false, message: '', title: ''})}
        title={alertOption.title}
        message={alertOption.message}
      />
    </>
  );
} 