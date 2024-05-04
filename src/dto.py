from pydantic import BaseModel
from typing import List

class reportRequestDto(BaseModel):
    factors: List[str]
    destPoint: str
    timeRange: str
    sex: bool
    age: int
    workDay: bool
    arrivalTime: int

class pointInfo(BaseModel):
    name: str
    matchRate: str
    rank: int

class timeRangeGroup(BaseModel):
    timeRange: str
    pointInfo: List[pointInfo]

class reportResponseDto(BaseModel):
    pointList: List[timeRangeGroup]