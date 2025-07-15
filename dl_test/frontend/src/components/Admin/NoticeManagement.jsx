import { useState, useEffect } from "react";
import api from "@/api/api";

export default function NoticeManagement() {
  const [notices, setNotices] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [newNotice, setNewNotice] = useState({ title: "", content: "" });
  const [editId, setEditId] = useState(null); // 수정 중인 공지 ID

  // 공지사항 목록 불러오기
  const fetchNotices = async () => {
    try {
      const res = await api.get("/admin/notice/select");
      setNotices(res.data);
    } catch (error) {
      console.error("공지사항 불러오기 실패:", error);
    }
  };

  useEffect(() => {
    fetchNotices();
  }, []);

  // 공지사항 등록 또는 수정
  const handleSubmitNotice = async () => {
    try {
      if (editId) {
        // 수정 요청
        await api.put(`/admin/notice/update/${editId}`, newNotice);
      } else {
        // 등록 요청
        await api.post("/admin/notice/register", newNotice);
      }

      setShowModal(false);
      setNewNotice({ title: "", content: "" });
      setEditId(null);
      fetchNotices(); // 목록 갱신
    } catch (error) {
      console.error("공지사항 저장 실패:", error);
    }
  };

  // 수정 버튼 클릭 시
  const handleEdit = (id) => {
    const target = notices.find((n) => n.id === id);
    if (target) {
      setNewNotice({ title: target.title, content: target.content });
      setEditId(id);
      setShowModal(true);
    }
  };

  // 삭제 버튼 클릭 시
  const handleDelete = async (id) => {
    if (confirm("정말 삭제하시겠습니까?")) {
      try {
        await api.delete(`/admin/notice/delete/${id}`);
        fetchNotices(); // 목록 갱신
      } catch (error) {
        console.error("공지사항 삭제 실패:", error);
      }
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6 text-center">공지사항 목록</h1>

      {/* 등록 버튼 우측 정렬 */}
      <div className="flex justify-end mb-4">
        <button
          onClick={() => {
            setShowModal(true);
            setEditId(null);
            setNewNotice({ title: "", content: "" });
          }}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          공지사항 작성
        </button>
      </div>

      {/* 테이블 */}
      <table className="min-w-full border border-gray-300 table-fixed">
        <thead>
          <tr className="bg-gray-100">
            <th className="py-2 px-4 border w-1/6">제목</th>
            <th className="py-2 px-4 border w-2/5">내용</th>
            <th className="py-2 px-4 border w-1/6 whitespace-nowrap">처리</th>
          </tr>
        </thead>
          <tbody>
            {notices.length === 0 ? (
              <tr>
                <td colSpan={5} className="py-4 text-center text-gray-500">
                  공지사항이 없습니다.
                </td>
              </tr>
            ) : (
            notices.map((notice) => (
              <tr key={notice.id} className="hover:bg-gray-50">
                <td className="py-2 px-4 border text-center truncate">{notice.title}</td>
                <td className="py-2 px-4 border text-center">{notice.content}</td>
                <td className="py-2 px-4 border text-center whitespace-nowrap">
                  <button
                    onClick={() => handleEdit(notice.id)}
                    className="px-3 py-1 text-white bg-blue-500 rounded hover:bg-blue-600 text-sm mr-2"
                  >
                    수정
                  </button>
            
                  <button
                    onClick={() => handleDelete(notice.id)}
                    className="px-3 py-1 text-white bg-red-500 rounded hover:bg-red-600 text-sm"
                  >
                    삭제
                  </button>
                </td>
              </tr>
            ))
            )}
          </tbody>
      </table>

      {/* 모달 */}
      {showModal && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-40 z-50">
          <div className="bg-white rounded-lg p-6 w-[90%] max-w-md shadow-lg">
            <h2 className="text-xl font-bold mb-4">
              {editId ? "공지사항 수정" : "공지사항 작성"}
            </h2>
            <input
              type="text"
              placeholder="제목"
              value={newNotice.title}
              onChange={(e) => setNewNotice({ ...newNotice, title: e.target.value })}
              className="w-full p-2 mb-3 border rounded"
            />
            <textarea
              placeholder="내용"
              value={newNotice.content}
              onChange={(e) => setNewNotice({ ...newNotice, content: e.target.value })}
              className="w-full p-2 mb-4 border rounded h-32 resize-none"
            ></textarea>
            <div className="flex justify-end space-x-2">
              <button
                onClick={handleSubmitNotice}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                {editId ? "수정 완료" : "등록"}
              </button>
                            <button
                onClick={() => {
                  setShowModal(false);
                  setEditId(null);
                  setNewNotice({ title: "", content: "" });
                }}
                className="px-4 py-2 bg-gray-400 text-white rounded hover:bg-gray-500"
              >
                취소
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
