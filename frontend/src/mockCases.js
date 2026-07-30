// 아래 4개 이미지는 전부 Wikimedia Commons의 CC 라이선스 병리 이미지입니다.
// 실제 환자/케이스 사진이 아니라 mock 데이터 표시용 참고 이미지입니다.
// - Normal_breast_histology.png (CC BY 4.0)
// - Histology_of_lactating_breast,_annotated.png (Public Domain Mark)
// - Micrograph_of_invasive_ductal_carcinoma_with_marked_nuclear_pleomorphism.jpg (CC BY 3.0)
// - Invasive_Ductal_Carcinoma_40x.jpg (CC BY-SA 3.0)
const THUMB_1 =
  "https://commons.wikimedia.org/wiki/Special:FilePath/Normal_breast_histology.png";
const THUMB_2 =
  "https://commons.wikimedia.org/wiki/Special:FilePath/Histology_of_lactating_breast,_annotated.png";
const THUMB_3 =
  "https://commons.wikimedia.org/wiki/Special:FilePath/Micrograph_of_invasive_ductal_carcinoma_with_marked_nuclear_pleomorphism.jpg";
const THUMB_4 =
  "https://commons.wikimedia.org/wiki/Special:FilePath/Invasive_Ductal_Carcinoma_40x.jpg";

export const MOCK_CASES = [
  { id: "C-0233", uploadedAt: "2026-07-21 09:33", status: "완료", stil: 47, abstain: true, abstainReason: "예측구간 폭 과다", hospital: "서울성모병원", thumbnail: THUMB_1 },
  { id: "C-0232", uploadedAt: "2026-07-21 09:12", status: "완료", stil: 22, abstain: false, abstainReason: null, hospital: "건양대병원", thumbnail: THUMB_2 },
  { id: "C-0231", uploadedAt: "2026-07-21 08:55", status: "완료", stil: 61, abstain: true, abstainReason: "컷오프 근접(50%±5)", hospital: "세브란스병원", thumbnail: THUMB_3 },
  { id: "C-0230", uploadedAt: "2026-07-21 08:40", status: "처리중", stil: null, abstain: null, abstainReason: null, hospital: "건양대병원", thumbnail: THUMB_4 },
  { id: "C-0229", uploadedAt: "2026-07-20 17:02", status: "완료", stil: 15, abstain: false, abstainReason: null, hospital: "서울성모병원", thumbnail: THUMB_1 },
];