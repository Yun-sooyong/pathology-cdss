import { useState } from 'react';
import { createBatch, uploadToSignedUrl, startBatch, uploadBatchLegacy } from './api';
import DashboardScreen from './screens/DashboardScreen';
import UploadScreen from './screens/UploadScreen';
import LoadingScreen from './screens/LoadingScreen';
import ErrorScreen from './screens/ErrorScreen';
import SummaryScreen from './screens/SummaryScreen';
import DetailScreen from './screens/DetailScreen';

function App() {
  const [screen, setScreen] = useState('dashboard');
  const [batchFiles, setBatchFiles] = useState([]);
  const [batchId, setBatchId] = useState(null);
  const [completedResults, setCompletedResults] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({});

  function handleNewAnalysis() {
    setScreen('upload');
  }

  // 지금은 케이스별로 실제 데이터를 따로 불러오는 기능이 없어서,
  // 어떤 케이스를 눌러도 일단 요약 화면(mock)으로 이동만 함.
  // 나중에 API 연동되면 caseId로 실제 데이터를 불러오도록 바꿔야 함.
  function handleViewCase(caseId) {
    console.log('케이스 상세 이동:', caseId);
    setScreen('summary');
  }

  async function handleAnalyze(files, model) {
    setIsUploading(true);
    setUploadProgress({});
    try {
      // ↓↓↓ 임시: /api/batches(signed URL 방식)가 아직 배포 전이라 legacy 방식으로 테스트 ↓↓↓
      const result = await uploadBatchLegacy(files, model);
      console.log('업로드 완료(legacy), batch_id:', result.batch_id);
      setBatchFiles(files);
      setBatchId(result.batch_id);
      setScreen('loading');

      /* signed URL 방식 준비되면 위 3줄 지우고 아래 주석 풀기:
      // 1단계: 배치 생성 — 파일명/크기만 먼저 보내고 서명된 업로드 URL을 받는다
      const { batch_id, uploads } = await createBatch(files, model);
      console.log('배치 생성 완료:', batch_id, uploads);

      // 2단계: 각 파일을 GCS에 직접 업로드 (Cloud Run을 거치지 않음, 병렬로 진행)
      // 주의: uploads 배열 순서가 files 배열 순서와 같다고 가정함 (서버에 순서 보장 확인 필요)
      await Promise.all(
        uploads.map((u, i) =>
          uploadToSignedUrl(u.upload_url, files[i].file, (percent) => {
            setUploadProgress((prev) => ({ ...prev, [files[i].name]: percent }));
          })
        )
      );

      // 3단계: 업로드 다 됐다는 신호 → 서버가 실제 분석을 큐에 등록
      await startBatch(batch_id);

      setBatchFiles(files);
      setBatchId(batch_id);
      setScreen('loading');
      */
    } catch (err) {
      console.error(err);
      alert(`업로드 중 오류가 발생했어요: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  }

  function handleCancelAll() {
    setScreen('summary');
  }

  function handleComplete(batch) {
    setCompletedResults(batch);
    setScreen('summary');
  }

  function handleViewDetail() {
    setScreen('detail');
  }

  function handleViewErrors(batch) {
    if (batch) setCompletedResults(batch);
    setScreen('error');
  }

  function handleGenerateReport() {
    alert('오류리포트 생성 (실제 PDF/문서 생성은 백엔드 연동 후 구현 예정)');
  }

  function handleBackToUpload() {
    setScreen('upload');
  }

  function handleBackToSummary() {
    setScreen('summary');
  }

  return (
    <div style={{ maxWidth: 1200, margin: '32px auto', padding: '0 24px' }}>
      {screen === 'dashboard' && (
        <DashboardScreen onNewAnalysis={handleNewAnalysis} onViewCase={handleViewCase} />
      )}

      {screen === 'upload' && (
        <UploadScreen onAnalyze={handleAnalyze} isUploading={isUploading} uploadProgress={uploadProgress} />
      )}

      {screen === 'loading' && (
        <LoadingScreen
          batchId={batchId}
          files={batchFiles}
          onCancelAll={handleCancelAll}
          onComplete={handleComplete}
          onViewErrors={handleViewErrors}
        />
      )}

      {screen === 'error' && (
        <ErrorScreen batch={completedResults} files={batchFiles} onGenerateReport={handleGenerateReport} onBackToUpload={handleBackToUpload} />
      )}

      {screen === 'summary' && (
        <SummaryScreen
          results={completedResults}
          files={batchFiles}
          onNewAnalysis={handleNewAnalysis}
          onViewDetail={handleViewDetail}
          onViewErrorReport={handleViewErrors}
        />
      )}

      {screen === 'detail' && <DetailScreen onBackToSummary={handleBackToSummary} />}
    </div>
  );
}

export default App;