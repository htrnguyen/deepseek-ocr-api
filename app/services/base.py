from abc import ABC, abstractmethod
from typing import Any


class BaseService(ABC):
    @abstractmethod
    async def process(self, *args, **kwargs) -> dict:
        pass

    def _build_response(self, data: dict) -> dict:
        return {"success": True, **data, "status": "success"}
