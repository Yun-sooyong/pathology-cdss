from pydantic import BaseModel


class PredictResponse(BaseModel):
    width: int
    height: int
    class_names: list[str]              # ["tumor","stroma","immune","necrosis","other"], 채널 순서
    class_map: list[list[int]]          # (H,W), 각 픽셀의 클래스 인덱스 (0~4, class_names 기준)
    class_pixel_ratio: dict[str, float]  # 클래스별 전체 픽셀 대비 비율(%)
    til_ratio_percent: float | None      # immune / (immune+stroma) * 100 (기획서 공식 sTIL)


class HealthResponse(BaseModel):
    status: str
    device: str
