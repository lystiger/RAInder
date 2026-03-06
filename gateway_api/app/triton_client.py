from dataclasses import dataclass


@dataclass
class TritonResult:
    upscaled_image: bytes
    width: int
    height: int
    compute_latency_ms: float


class TritonClient:
    """Placeholder gRPC client.

    Next step: generate grpc stubs from proto and implement Process RPC call.
    """

    def upscale(self, raw_image: bytes, model_name: str, scale_factor: float) -> TritonResult:
        raise NotImplementedError("Triton gRPC client is not implemented yet.")
