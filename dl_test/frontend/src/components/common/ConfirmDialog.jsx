import {
  Button,
  Dialog,
  DialogHeader,
  DialogBody,
  DialogFooter,
} from "@material-tailwind/react";

export default function ConfirmDialog({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
}) {
  // Dialog가 열려있지 않으면 아무것도 렌더링하지 않습니다.
  if (!isOpen) return null;

  return (
    <Dialog open={isOpen} handler={onClose} size="sm">
      <DialogHeader>{title}</DialogHeader>
      <DialogBody divider>
        <p className="text-base font-normal text-gray-600">
          {message}
        </p>
      </DialogBody>
      <DialogFooter className="space-x-2">
      <Button variant="text" className="bg-gray-400 hover:bg-gray-200 text-white" 
          onClick={onClose}>
          취소
        </Button>
        <Button variant="text" className="bg-blue-700 hover:bg-blue-300 text-white" 
          onClick={onConfirm}>
          확인
        </Button>
      </DialogFooter>
    </Dialog>
  );
} 