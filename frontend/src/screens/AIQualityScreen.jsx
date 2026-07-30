import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Trophy } from "lucide-react";

// 프로젝트 연구 단계에서 확정된 "최종 3축 종합표" (BCSS_파이프라인_상세, Phase4_5_최종정리 문서 기준)
// 실시간 API 데이터가 아니라, 연구 발표에 쓰인 고정된 검증 결과입니다.
const FINAL_COMPARISON = [
  { model: "U-Net", dice: 0.657, tilError: 6.09, misclass: "9/151", misclassPct: 5.96, oodCoverage: 80.0 },
  { model: "DeepLabV3+", dice: 0.667, tilError: 7.02, misclass: "14/151", misclassPct: 9.27, oodCoverage: 80.0 },
  { model: "FPN", dice: 0.657, tilError: 5.79, misclass: "14/151", misclassPct: 9.27, oodCoverage: 84.4 },
];

function bestModel(key, direction) {
  return FINAL_COMPARISON.reduce((best, cur) =>
    direction === "max" ? (cur[key] > best[key] ? cur : best) : (cur[key] < best[key] ? cur : best)
  ).model;
}

const bestDice = bestModel("dice", "max");
const bestTilError = bestModel("tilError", "min");
const bestMisclass = bestModel("misclassPct", "min");
const bestCoverage = bestModel("oodCoverage", "max");

function WinnerBadge({ model, winner }) {
  if (model !== winner) return null;
  return (
    <Badge className="ml-1.5 border border-seg-abstain/40 bg-seg-abstain/10 text-seg-abstain">
      <Trophy size={10} className="mr-1" /> 1위
    </Badge>
  );
}

export default function AIQualityScreen() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-xl font-semibold">AI 모델 품질 통계</h1>
        <p className="mt-1 text-xs text-muted-foreground">
          연구 단계에서 검증된 3개 세그멘테이션 모델 비교 결과 (BCSS 데이터셋, 실시간 데이터 아님)
        </p>
      </div>

      <Card className="mb-6 border border-seg-abstain/30 bg-seg-abstain/10">
        <CardContent className="pt-5">
          <div className="text-sm font-semibold text-seg-abstain">
            핵심 결론: Dice 1등 모델이 임상 결정에서는 최하위권
          </div>
          <p className="mt-2 text-sm text-muted-foreground">
            DeepLabV3+가 Dice 점수는 가장 높지만(0.667), 실제 50% 컷오프 기준 오분류는
            9.27%로 세 모델 중 가장 나쁩니다. 반대로 U-Net은 Dice가 가장 낮은데도
            임상 오분류율은 가장 낮습니다(5.96%). Dice 점수만으로 모델을 선택하면
            안 된다는 이 프로젝트의 핵심 주장을 뒷받침하는 결과입니다.
          </p>
        </CardContent>
      </Card>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>최종 3축 종합 비교</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>모델</TableHead>
                <TableHead>macro Dice ↑</TableHead>
                <TableHead>sTIL 절대오차 ↓</TableHead>
                <TableHead>컷오프 오분류 ↓</TableHead>
                <TableHead>OOD 커버리지</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {FINAL_COMPARISON.map((row) => (
                <TableRow key={row.model}>
                  <TableCell className="font-semibold">{row.model}</TableCell>
                  <TableCell>
                    {row.dice.toFixed(3)}
                    <WinnerBadge model={row.model} winner={bestDice} />
                  </TableCell>
                  <TableCell>
                    {row.tilError.toFixed(2)}%p
                    <WinnerBadge model={row.model} winner={bestTilError} />
                  </TableCell>
                  <TableCell>
                    {row.misclass} ({row.misclassPct.toFixed(2)}%)
                    <WinnerBadge model={row.model} winner={bestMisclass} />
                  </TableCell>
                  <TableCell>
                    {row.oodCoverage.toFixed(1)}%
                    <WinnerBadge model={row.model} winner={bestCoverage} />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <p className="mt-3 text-[11px] text-muted-foreground">
            OOD 커버리지는 held-out 45개 기관 pooled 기준 (목표 90% 대비 80~84.4%,
            기권율과 함께 해석 필요 — 기권율이 같이 오르면 "안전한 실패"로 판단)
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>참고 — 지표별 최선 모델이 다름</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
            <div>
              <div className="text-xs text-muted-foreground">Dice 최고</div>
              <div className="font-semibold">{bestDice}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">sTIL 오차 최소</div>
              <div className="font-semibold">{bestTilError}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">임상 오분류 최소</div>
              <div className="font-semibold">{bestMisclass}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">OOD 커버리지 최고</div>
              <div className="font-semibold">{bestCoverage}</div>
            </div>
          </div>
          <p className="mt-3 text-[11px] text-muted-foreground">
            어떤 단일 지표로도 "최고 모델"을 정할 수 없다는 것이 이 비교의 결론입니다.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}