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
    PlanTransform,
    SampleOut,
    Sampler,
    SamplerPlugin,
    SchedulePolicy,
    TargetView,
    Verifier,
)
from .frontend import (
    Abort,
    EngineClient,
    EngineRequest,
    FrontendBundle,
    FrontendPlugin,
    Logprobs,
    Parser,
    TokenDelta,
    Tokenizer,
)
from .manifest import SLOTS, Cardinality, Manifest, Provenance, Resource, Tunable
from .memory import Ledger, Pool
from .mixer import Mixer, Planner
from .model import AuxSpec, BuildCtx, LayerDecl, ModelPlugin, ModelSpec, Unit, WeightMap, WeightRule, WeightSource
from .ops import Constraint, KernelRegistry, KernelSpec, Plugin, Registries, Registry
from .plan import Binding, Lock, Plan, UnitPin
from .platform import Fingerprint, Platform
from .quant import Converter, Dialect, Layout, QTensor, Scale, Scheme, SchemeImpl
from .state import (
    HitLen,
    InStepPublish,
    MemOp,
    Paged,
    Pages,
    Payload,
    PrefixView,
    Rewindable,
    SeqView,
    Slot,
    StateKind,
    StateKindFactory,
    StateSpec,
)
from .telemetry import Event
from .terms import Ref, canonical_hash, parse_versioned, to_term

__all__ = [
    "SLOTS", "Abort", "AuxSpec", "Batch", "BatchPlan", "Binding", "BuildCtx", "Cardinality", "Constraint",
    "Converter", "DeviceBatch", "Dialect", "DraftPolicy", "Drafter", "DrafterPlugin", "EngineClient",
    "EngineRequest", "Event", "Fingerprint", "FrontendBundle", "FrontendPlugin", "HitLen", "InStepPublish",
    "KernelRegistry", "KernelSpec", "LayerDecl", "Layout", "Ledger", "Lock", "LogitsProcessor", "Logprobs",
    "Manifest", "MemOp", "Mixer", "ModelPlugin", "ModelSpec", "Paged", "Pages", "Parser", "Payload", "Plan",
    "PlanTransform", "Planner", "Platform", "Plugin", "Pool", "PrefixView", "Provenance", "QTensor", "Ref",
    "Registries", "Registry", "Resource", "Rewindable", "SampleOut", "Sampler", "SamplerPlugin", "SamplingParams",
    "Scale", "SchedulePolicy", "Scheme", "SchemeImpl", "Seq", "SeqTable", "SeqView", "ShapeKey", "Slot", "StateKind",
    "StateKindFactory", "StateSpec", "Status", "StepCtx", "TargetView", "TokenDelta", "TokenSrc", "Tokenizer",
    "Tunable", "Unit", "UnitPin", "Verifier", "WeightMap", "WeightRule", "WeightSource", "canonical_hash",
    "parse_versioned", "sequence_seed", "to_term",
]
