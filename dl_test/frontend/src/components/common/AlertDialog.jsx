import {
  Button,
  Dialog,
  DialogHeader,
  DialogBody,
  DialogFooter,
} from "@material-tailwind/react";

export default function AlertDialog({
  isOpen,
  onClose,
  message,
  title,
}) {
  // Dialog가 열려있지 않으면 아무것도 렌더링하지 않습니다.
  if (!isOpen) return null;

  // HTML 태그가 포함되어 있는지 확인하는 함수
  const isHtml = (str) => {
    return /<[a-z][\s\S]*>/i.test(str);
  };

  return (
    <Dialog open={isOpen} handler={onClose} size="xs">
      <DialogHeader>{title}</DialogHeader>
      <DialogBody divider>
        {isHtml(message) ? (
          <div 
            className="text-base font-normal text-gray-600"
            dangerouslySetInnerHTML={{ __html: message }}
          />
        ) : (
          <p className="text-base font-normal text-gray-600">
            {message}
          </p>
        )}
      </DialogBody>
      <DialogFooter className="space-x-2">
        <Button variant="text" className="bg-gray-500 hover:bg-gray-600 text-white" onClick={onClose}>
          닫기
        </Button>
      </DialogFooter>
    </Dialog>
  );
} 