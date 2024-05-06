from pydantic import BaseModel
from typing import List

class ReportRequestDto(BaseModel):
    factors: List[str]
    station: str
    destPoint: str
    timeRange: str
    sex: bool
    age: int
    workDay: bool
    arrivalTime: int

class PointInfo(BaseModel):
    name: str
    matchRate: str
    rank: int

class TimeRangeGroup(BaseModel):
    timeRange: str
    pointInfo: List[PointInfo]

class ReportResponseDto(BaseModel):
    pointList: List[TimeRangeGroup]