import NoticeGetNewInfo from "./NoticeGetNewInfo";
import NoticeManagement from "./NoticeManagement";

export default function Notice() {
  return (
    <div className="flex flex-col gap-12 p-6">
      <NoticeGetNewInfo />
      <NoticeManagement />
    </div>
  );
}
