from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class Status(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


# Cost per 1000 tokens (input, output) in USD
COST_TABLE: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o":              (0.005,  0.015),
    "gpt-4o-mini":         (0.00015, 0.0006),
    "gpt-4-turbo":         (0.01,   0.03),
    "gpt-3.5-turbo":       (0.0005, 0.0015),
    # Anthropic
    "claude-opus-4-5":     (0.015,  0.075),
    "claude-sonnet-4-5":   (0.003,  0.015),
    "claude-haiku-4-5":    (0.00025, 0.00125),
    # Ollama — local, no cost
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for a call. Returns 0.0 for unknown/local models."""
    rates = COST_TABLE.get(model)
    if not rates:
        return 0.0
    input_cost  = (input_tokens  / 1000) * rates[0]
    output_cost = (output_tokens / 1000) * rates[1]
    return round(input_cost + output_cost, 8)


@dataclass
class Trace:
    """A single captured LLM API call."""

    # Identity
    id:            str      = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:     datetime = field(default_factory=datetime.utcnow)

    # Call metadata
    provider:      Provider = Provider.OPENAI
    model:         str      = ""
    status:        Status   = Status.SUCCESS

    # Token usage
    input_tokens:  int      = 0
    output_tokens: int      = 0

    # Performance
    latency_ms:    float    = 0.0   # wall-clock time for the API call

    # Cost
    cost_usd:      float    = 0.0   # estimated, auto-filled if not provided

    # Error info (populated on status=ERROR)
    error_type:    Optional[str] = None
    error_message: Optional[str] = None

    # Optional context tags (e.g. {"feature": "search", "user_id": "u123"})
    tags:          dict = field(default_factory=dict)

    def __post_init__(self):
        # Auto-estimate cost if not explicitly set
        if self.cost_usd == 0.0 and self.status == Status.SUCCESS:
            self.cost_usd = estimate_cost(
                self.model, self.input_tokens, self.output_tokens
            )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {
            "id":             self.id,
            "timestamp":      self.timestamp.isoformat(),
            "provider":       self.provider.value,
            "model":          self.model,
            "status":         self.status.value,
            "input_tokens":   self.input_tokens,
            "output_tokens":  self.output_tokens,
            "total_tokens":   self.total_tokens,
            "latency_ms":     self.latency_ms,
            "cost_usd":       self.cost_usd,
            "error_type":     self.error_type,
            "error_message":  self.error_message,
            "tags":           self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trace":
        data = dict(data)
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["provider"]  = Provider(data["provider"])
        data["status"]    = Status(data["status"])
        data.pop("total_tokens", None)   # computed property, not stored
        return cls(**data)


@dataclass
class AggregatedStats:
    """Summary stats over a collection of Traces — used by CLI and dashboard."""

    total_calls:     int   = 0
    success_calls:   int   = 0
    error_calls:     int   = 0

    total_tokens:    int   = 0
    total_input:     int   = 0
    total_output:    int   = 0

    total_cost_usd:  float = 0.0

    avg_latency_ms:  float = 0.0
    p50_latency_ms:  float = 0.0
    p95_latency_ms:  float = 0.0

    # Breakdowns keyed by provider/model name
    by_provider:     dict  = field(default_factory=dict)
    by_model:        dict  = field(default_factory=dict)

    @property
    def error_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return round(self.error_calls / self.total_calls, 4)

    @classmethod
    def from_traces(cls, traces: list[Trace]) -> "AggregatedStats":
        if not traces:
            return cls()

        import statistics

        latencies = [t.latency_ms for t in traces]
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)

        by_provider: dict = {}
        by_model:    dict = {}

        for t in traces:
            p = t.provider.value
            m = t.model
            for bucket, key in [(by_provider, p), (by_model, m)]:
                if key not in bucket:
                    bucket[key] = {
                        "calls": 0, "tokens": 0,
                        "cost_usd": 0.0, "errors": 0,
                    }
                bucket[key]["calls"]    += 1
                bucket[key]["tokens"]   += t.total_tokens
                bucket[key]["cost_usd"] += t.cost_usd
                if t.status == Status.ERROR:
                    bucket[key]["errors"] += 1

        return cls(
            total_calls    = n,
            success_calls  = sum(1 for t in traces if t.status == Status.SUCCESS),
            error_calls    = sum(1 for t in traces if t.status == Status.ERROR),
            total_tokens   = sum(t.total_tokens  for t in traces),
            total_input    = sum(t.input_tokens  for t in traces),
            total_output   = sum(t.output_tokens for t in traces),
            total_cost_usd = round(sum(t.cost_usd for t in traces), 6),
            avg_latency_ms = round(statistics.mean(latencies), 2),
            p50_latency_ms = round(sorted_lat[int(n * 0.50)], 2),
            p95_latency_ms = round(sorted_lat[int(n * 0.95)], 2),
            by_provider    = by_provider,
            by_model       = by_model,
        )