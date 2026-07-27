import { useState } from "react";
import TopNav from "./components/TopNav";
import { Routes, Route, useNavigate, useParams } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import CaseBoardScreen from "./screens/CaseBoardScreen";
import DashboardScreen from "./screens/DashboardScreen";
import UploadScreen from "./screens/UploadScreen";
import LoadingScreen from "./screens/LoadingScreen";
import ErrorScreen from "./screens/ErrorScreen";
import SummaryScreen from "./screens/SummaryScreen";
import DetailScreen from "./screens/DetailScreen";
import { MOCK_CASES } from "./mockCases";
import { useLocation } from "react-router-dom";
import AIQualityScreen from "./screens/AIQualityScreen";
import {
  createBatch,
  uploadToSignedUrl,
  startBatch,
  uploadBatchLegacy,
  getJobDetail,
} from "./api";

function App() {
  const navigate = useNavigate();

  const [batchFiles, setBatchFiles] = useState([]);
  const [batchId, setBatchId] = useState(null);
  const [completedResults, setCompletedResults] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({});

  const pendingQueue = MOCK_CASES.filter((c) => c.abstain === true).map(
    (c) => c.id,
  );

  function handleNewAnalysis() {
    navigate("/upload");
  }

  // caseId로 실제 데이터를 불러오는 API가 아직 없어서 mock 케이스 목록 기준으로 detail 화면으로 바로 이동.
  function handleViewCase(caseId, mockThumbnail) {
    navigate(`/detail/${caseId}`, { state: { mockThumbnail } });
  }
async function handleComplete(batch) {
  const jobs = batch?.jobs ?? [];
  try {
    const detailedJobs = await Promise.all(
      jobs.map((j) => getJobDetail(j.job_id))
    );
    setCompletedResults(detailedJobs);
  } catch (err) {
    console.error("job 상세 조회 실패:", err);
    setCompletedResults(jobs);
  }
  navigate("/summary");
}
  async function handleAnalyze(files, model) {
    setIsUploading(true);
    setUploadProgress({});
    try {
      // ↓↓↓ 임시: /api/batches(signed URL 방식)가 아직 배포 전이라 legacy 방식으로 테스트 ↓↓↓
      const result = await uploadBatchLegacy(files, model);
      console.log("업로드 완료(legacy), batch_id:", result.batch_id);
      setBatchFiles(files);
      setBatchId(result.batch_id);
      navigate("/loading");

      /* signed URL 방식 준비되면 위 3줄 지우고 아래 주석 풀기:
      const { batch_id, uploads } = await createBatch(files, model);
      await Promise.all(
        uploads.map((u, i) =>
          uploadToSignedUrl(u.upload_url, files[i].file, (percent) => {
            setUploadProgress((prev) => ({ ...prev, [files[i].name]: percent }));
          })
        )
      );
      await startBatch(batch_id);
      setBatchFiles(files);
      setBatchId(batch_id);
      navigate('/loading');
      */
    } catch (err) {
      console.error(err);
      alert(`업로드 중 오류가 발생했어요: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  }

function handleCancelAll() {
    navigate("/summary");
  }

 

  function handleViewDetail() {
    navigate("/detail");
  }
  function handleViewErrors(batch) {
    if (batch) setCompletedResults(batch);
    navigate("/error");
  }

  function handleGenerateReport() {
    alert("오류리포트 생성 (실제 PDF/문서 생성은 백엔드 연동 후 구현 예정)");
  }

  function handleBackToUpload() {
    navigate("/upload");
  }

  function handleBackToSummary() {
    navigate("/summary");
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 overflow-y-auto px-6 py-8">
        <TopNav />
        <Routes>
          <Route
            path="/"
            element={
              <DashboardScreen
                onNewAnalysis={handleNewAnalysis}
                onViewCase={handleViewCase}
              />
            }
          />
          <Route path="/quality" element={<AIQualityScreen />} />
          <Route
            path="/upload"
            element={
              <UploadScreen
                onAnalyze={handleAnalyze}
                isUploading={isUploading}
                uploadProgress={uploadProgress}
              />
            }
          />
          <Route
            path="/loading"
            element={
              <LoadingScreen
                batchId={batchId}
                files={batchFiles}
                onCancelAll={handleCancelAll}
                onComplete={handleComplete}
                onViewErrors={handleViewErrors}
              />
            }
          />
          <Route
            path="/board"
            element={<CaseBoardScreen onViewCase={handleViewCase} />}
          />
          <Route
            path="/error"
            element={
              <ErrorScreen
                batch={completedResults}
                files={batchFiles}
                onGenerateReport={handleGenerateReport}
                onBackToUpload={handleBackToUpload}
              />
            }
          />
          <Route
            path="/summary"
            element={
              <SummaryScreen
                results={completedResults}
                files={batchFiles}
                onNewAnalysis={handleNewAnalysis}
                onViewDetail={handleViewDetail}
                onViewErrorReport={handleViewErrors}
              />
            }
          />
          <Route
            path="/detail/:caseId?"
            element={
              <DetailRoute
                pendingQueue={pendingQueue}
                onBackToSummary={handleBackToSummary}
              />
            }
          />
        </Routes>
      </main>
    </div>
  );
}

// URL 파라미터(caseId)를 DetailScreen에 연결하고, "다음 케이스" 이동도 URL 변경으로 처리
function DetailRoute({ pendingQueue, onBackToSummary }) {
  const { caseId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const realJob = location.state?.job ?? null;
  const mockThumbnail = location.state?.mockThumbnail ?? null;

  function handleNextPendingCase() {
    const idx = pendingQueue.indexOf(caseId);
    const next = pendingQueue[idx + 1];
    if (next) navigate(`/detail/${next}`);
  }

  return (
    <DetailScreen
      caseId={caseId}
      realJob={realJob}
      mockThumbnail={mockThumbnail}
      pendingQueue={pendingQueue}
      onNextPendingCase={handleNextPendingCase}
      onBackToSummary={onBackToSummary}
    />
  );
}

export default App;
