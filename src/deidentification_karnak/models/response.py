from pydantic import BaseModel, Field


class MaskGroup(BaseModel):
    station_name: str = Field("*", alias="stationName")
    color: str = Field(..., description="Background color as hex RRGGBB (no leading #)")
    rectangles: list[str] = Field(
        ..., description='Rectangles in "x y width height" format'
    )

    model_config = {"populate_by_name": True}


class DeidentificationResponse(BaseModel):
    masks: list[MaskGroup] | None = None
    message: str
    sop_instance_uid: str | None = None
