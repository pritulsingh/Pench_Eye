from pydantic import BaseModel, ConfigDict, computed_field
from typing import Generic, TypeVar, List, Optional
from enum import Enum

T = TypeVar('T')


class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    size: int
    model_config = ConfigDict(from_attributes=True)

    @computed_field
    @property
    def pages(self) -> int:
        if self.size <= 0:
            return 0
        return max(1, -(-self.total // self.size))


class ErrorResponse(BaseModel):
    detail: str


class SuccessResponse(BaseModel):
    success: bool
    message: Optional[str] = None


class StatusEnum(str, Enum):
    SUCCESS = "success"
    ERROR = "error"