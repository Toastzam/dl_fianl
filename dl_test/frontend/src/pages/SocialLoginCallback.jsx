import { useEffect } from "react";
import { useNavigate } from "react-router-dom";

const SocialLoginCallback = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");
    const userId = params.get("userId");
    const name = params.get("name");

    if (token && userId) {
      localStorage.setItem("token", token);
      localStorage.setItem("userId", userId);
      localStorage.setItem("isLoggedIn", "true");
      if (name) {
        localStorage.setItem("name", name);
      }
      window.localStorage.setItem('logout', Date.now()); // 멀티탭 동기화
      alert("소셜 로그인 성공!");
      window.location.replace("/");
    } else {
      alert("소셜 로그인 실패 또는 정보 없음");
      navigate("/login");
    }
  }, [navigate]);

  return <div>로그인 처리 중...</div>;
};

export default SocialLoginCallback;
