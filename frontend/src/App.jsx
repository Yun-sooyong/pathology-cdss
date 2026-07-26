import { useState } from 'react';
import { uploadBatch, getResult } from './api';
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
  const [completedResults, setCompletedResults] = useState([]);

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
    try {
      // 파일 전부를 한 번에 담아서 업로드 → batch_id 하나를 받는다
      const result = await uploadBatch(files, model);
      console.log('업로드 완료, batch_id:', result.batch_id);
      setBatchFiles(files);
      setBatchId(result.batch_id);
      setScreen('loading');
    } catch (err) {
      console.error(err);
      alert(`업로드 중 오류가 발생했어요: ${err.message}`);
    }
  }

  function handleCancelAll() {
    setScreen('summary');
  }

  function handleComplete(results) {
    setCompletedResults(results);
    setScreen('summary');
  }

  function handleViewDetail() {
    setScreen('detail');
  }

  function handleViewErrors() {
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

      {screen === 'upload' && <UploadScreen onAnalyze={handleAnalyze} />}

      {screen === 'loading' && (
        <LoadingScreen
          batchId={batchId}
          onCancelAll={handleCancelAll}
          onComplete={handleComplete}
          onViewErrors={handleViewErrors}
        />
      )}

      {screen === 'error' && (
        <ErrorScreen onGenerateReport={handleGenerateReport} onBackToUpload={handleBackToUpload} />
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