import React from 'react';
import { Typography, Button } from "@material-tailwind/react";
import ColorPreview from './pub/ColorPreview.jsx';

const HeaderRightControls = ({
  isLoggedIn,
  userName,
  userId,
  NODE_ENV,
  onOpenSignup,
  onOpenLogin,
  onLogout,
  onOpenColors,
  setOpenSignup,
  setOpenLogin,
  setIsPubView
}) => {
  return (
    <>
      {!isLoggedIn ? (
        <>
          <Typography
            variant="small"
            className="text-blue-gray-600 hover:text-blue-500 cursor-pointer"
            onClick={onOpenSignup}
          >
            회원가입
          </Typography>
          <Typography
            variant="small"
            className="text-blue-gray-600 hover:text-blue-500 cursor-pointer"
            onClick={onOpenLogin}
          >
            로그인
          </Typography>
        </>
      ) : (
        <div className="flex items-center gap-4">
          <Typography
            variant="small"
            className="text-blue-gray-600"
          >
            {userName ? `${userName}님, 환영합니다!` : `${userId}님, 환영합니다!`}
          </Typography>
          <Button
            variant="outlined"
            className="text-blue-gray-600 border-blue-gray-300 hover:bg-blue-gray-100"
            onClick={onLogout}
          >
            로그아웃
          </Button>
        </div>
      )}
      {NODE_ENV === 'local' && (
        <Typography
          variant="small"
          className="text-blue-gray-600 hover:text-blue-500 cursor-pointer"
          onClick={onOpenColors}
        >
          Colors
        </Typography>
      )}
    </>
  );
};

export default HeaderRightControls;
