"""Plugin ABI: dataclasses and Protocols only (design §3). No runtime state and no device calls.

Importing `holon.api` must not import torch or numpy, so that model recipes stay pure data.
"""
from .batch import (
    Batch,
    BatchPlan,
    DeviceBatch,
    SamplingParams,
    Seq,
    SeqTable,
    ShapeKey,
    Status,
    StepCtx,
    TokenSrc,
    sequence_seed,
)
from .decoding import (
    Drafter,
    DrafterPlugin,
    DraftPolicy,
    LogitsProcessor,
    LogitsProcessorPlugin,
    MaskRow,
    PlanTransform,
    RequestError,
    SampleOut,
    Sampler,
    SamplerPlugin,
    SchedulePolicy,
    TargetView,
    Verifier,
    VocabView,
)
from .frontend import (
    ERROR_KINDS,
    FINISH_REASONS,
    Abort,
    EngineClient,
    EngineRequest,
    FrontendBundle,
    FrontendPlugin,
    Logprobs,
    ParseEvent,
    Reasoning,
    Text,
    TokenDelta,
    Tokenizer,
    ToolArgsDelta,
    ToolCallEnd,
    ToolCallStart,
)
from .hooks import Rewindable  # the optional hooks: `holon.api.hooks`
from .manifest import ABI, SLOTS, Cardinality, Manifest, Provenance, Resource, Tunable
from .memory import Ledger, Pool
from .mixer import Mixer, Planner
from .model import AuxSpec, BuildCtx, LayerDecl, ModelPlugin, ModelSpec, Unit, WeightMap, WeightRule, WeightSource
from .ops import KERNEL_PROPS, Constraint, KernelRegistry, KernelSpec, Plugin, Registries, Registry
from .plan import Binding, Lock, Plan, UnitPin
from .platform import Fingerprint, Platform
from .quant import Layout, QTensor, Scale, Scheme
from .state import (
    HitLen,
    InStepPublish,
    MemOp,
    Paged,
    Pages,
    Payload,
    PrefixView,
    SeqView,
    Slot,
    StateKind,
    StateKindFactory,
    StateSpec,
    rewind_modules,
)
from .telemetry import Event
from .terms import Ref, canonical_hash, parse_versioned, to_term

__all__ = [
    "ABI", "ERROR_KINDS", "FINISH_REASONS", "KERNEL_PROPS", "SLOTS", "Abort", "AuxSpec", "Batch", "BatchPlan",
    "Binding", "BuildCtx", "Cardinality", "Constraint", "DeviceBatch", "DraftPolicy", "Drafter", "DrafterPlugin",
    "EngineClient", "EngineRequest", "Event", "Fingerprint", "FrontendBundle", "FrontendPlugin", "HitLen",
    "InStepPublish", "KernelRegistry", "KernelSpec", "LayerDecl", "Layout", "Ledger", "Lock", "LogitsProcessor",
    "LogitsProcessorPlugin", "Logprobs", "Manifest", "MaskRow", "MemOp", "Mixer", "ModelPlugin", "ModelSpec", "Paged",
    "Pages", "ParseEvent", "Payload", "Plan", "PlanTransform", "Planner", "Platform", "Plugin", "Pool", "PrefixView",
    "Provenance", "QTensor", "Reasoning", "Ref", "Registries", "Registry", "RequestError", "Resource", "Rewindable",
    "SampleOut", "Sampler", "SamplerPlugin", "SamplingParams", "Scale", "SchedulePolicy", "Scheme", "Seq", "SeqTable",
    "SeqView", "ShapeKey", "Slot", "StateKind", "StateKindFactory", "StateSpec", "Status", "StepCtx", "TargetView",
    "Text", "TokenDelta", "TokenSrc", "Tokenizer", "ToolArgsDelta", "ToolCallEnd", "ToolCallStart", "Tunable", "Unit",
    "UnitPin", "Verifier", "VocabView", "WeightMap", "WeightRule", "WeightSource", "canonical_hash", "parse_versioned",
    "rewind_modules", "sequence_seed", "to_term",
]
