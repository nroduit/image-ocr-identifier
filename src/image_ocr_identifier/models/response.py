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


class ReportingResponse(BaseModel):
    detected_tags: list[str] = Field(
        default_factory=list,
        description="DICOM tag names whose values were detected in the image text",
    )
    message: str
    sop_instance_uid: str | None = None
