from typing import NamedTuple, List, Protocol, Any, runtime_checkable, Optional
from dataclasses import dataclass, asdict
from deepdiff import DeepHash

class Metadata(NamedTuple):
    n: int
    request_id: str


class PromptNamespace(NamedTuple):
    prompt: str
    namespace: str

@dataclass(frozen=True)
class Request:
    prompt: str
    n: int
    request_id: str
    namespace: str

    def __getattr__(self, name: str) -> None:
        return None

    def hash(self) -> int:
        """
        Defines the parameters that uniquely identify a request in the cache.
        """
        params = asdict(self)
        del params["request_id"]
        del params["namespace"]
        del params["n"]
        return DeepHash(params)[params]

class Batch(NamedTuple):
    requests: List[Request]

class Response(NamedTuple):
    data: List[Any]
    cached: Optional[List[bool]] = None
    duplicated: Optional[List[bool]] = None

@runtime_checkable
class SingleRequestModel(Protocol):
    """Model that processes one request at a time, can return multiple results."""

    async def request(self, request: Request) -> Response: ...


@runtime_checkable
class BatchRequestModel(Protocol):
    """Model that processes multiple requests at once, returns multiple results per request."""

    async def batch_request(self, batch: Batch) -> List[Response]: ...
