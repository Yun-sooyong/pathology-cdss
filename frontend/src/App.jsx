import { useState, useEffect } from "react";
import { Routes, Route, useNavigate, useParams, useLocation } from "react-router-dom";
import TopNav from "./components/TopNav";
import Sidebar from "./components/Sidebar";
import CaseBoardScreen from "./screens/CaseBoardScreen";
import DashboardScreen from "./screens/DashboardScreen";
import UploadScreen from "./screens/UploadScreen";
import LoadingScreen from "./screens/LoadingScreen";
import ErrorScreen from "./screens/ErrorScreen";
import SummaryScreen from "./screens/SummaryScreen";
import DetailScreen from "./screens/DetailScreen";
import AIQualityScreen from "./screens/AIQualityScreen";
import {
  createBatch,
  uploadToSignedUrl,
  startBatch,
  getJobDetail,
  getCases,
} from "./api";

function App() {
  const navigate = useNavigate();

  const [batchFiles, setBatchFiles] = useState([]);
  const [batchId, setBatchId] = useState(null);
  const [completedResults, setCompletedResults] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({});
  const [pendingQueue, setPendingQueue] = useState([]);

  // 기권 케이스 목록을 앱 시작 시 한 번 가져와서 "다음 확인 필요 케이스" 이동에 사용
  useEffect(() => {
    getCases({ limit: 50 })
      .then((data) => {
        const abstainIds = data.results
          .filter((c) => c.abstain === true)
          .map((c) => c.job_id);
        setPendingQueue(abstainIds);
      })
      .catch((err) => console.error("기권 케이스 목록 조회 실패:", err));
  }, []);

  function handleNewAnalysis() {
    navigate("/upload");
  }

  function handleViewCase(caseId) {
    navigate(`/detail/${caseId}`);
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
      navigate("/loading");
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
  const [realJob, setRealJob] = useState(location.state?.job ?? null);

  useEffect(() => {
    // Summary에서 이미 job 데이터를 들고 온 경우엔 다시 안 부름
    if (location.state?.job) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- location.state 변경 시 realJob 동기화, 의도된 동작
      setRealJob(location.state.job);
      return;
    }
    // 대시보드/케이스보드에서 caseId(job_id)만 갖고 들어온 경우 직접 조회
    if (caseId) {
      getJobDetail(caseId)
        .then(setRealJob)
        .catch((err) => {
          console.error("job 상세 조회 실패:", err);
          setRealJob(null);
        });
    }
  }, [caseId, location.state]);

  function handleNextPendingCase() {
    const idx = pendingQueue.indexOf(caseId);
    const next = pendingQueue[idx + 1];
    if (next) navigate(`/detail/${next}`);
  }

  return (
    <DetailScreen
      caseId={caseId}
      realJob={realJob}
      pendingQueue={pendingQueue}
      onNextPendingCase={handleNextPendingCase}
      onBackToSummary={onBackToSummary}
    />
  );
}

export default App;