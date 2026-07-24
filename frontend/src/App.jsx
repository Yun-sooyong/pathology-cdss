import { useState } from "react";
import UploadScreen from "./screens/UploadScreen";
import LoadingScreen from "./screens/LoadingScreen";

function App() {
  const [screen, setScreen] = useState("upload");

  function handleAnalyze(files, model) {
    console.log("분석 시작:", files, model);
    setScreen("loading");
  }

  function handleCancelAll() {
    setScreen("upload");
  }
  function handleCancelFile(fileName) {
    console.log("개별 취소:", fileName);
    alert(`${fileName} 작업을 취소했습니다 (임시 동작)`);
  }

  function handleComplete() {
    alert("결과 요약 화면은 아직 안 만들었어요 (다음 단계에서 만들 거예요)");
    // 나중에 SummaryScreen 만들면: setScreen('summary');
  }

  return (
    <div style={{ maxWidth: 1200, margin: "32px auto", padding: "0 24px" }}>
      {screen === "upload" && <UploadScreen onAnalyze={handleAnalyze} />}
      {screen === "loading" && (
        <LoadingScreen
          onCancelFile={handleCancelFile}
          onCancelAll={handleCancelAll}
          onComplete={handleComplete}
        />
      )}
    </div>
  );
}

export default App;
