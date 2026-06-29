"""Collaborative Agent Reasoning Ecosystem (CARE)."""

from importlib.metadata import PackageNotFoundError, version as _version

from care.bulk_import import (
    BulkImportEntry,
    BulkImportReport,
    ImportStatus,
    import_chains,
)
from care.capability_priming import (
    CapabilityPayload,
    CapabilityPrimingError,
    build_capability_payload,
)
from care.chain_export import (
    ChainExportError,
    ExportFormat,
    ExportResult,
    export_chain,
)
from care.conflict import (
    ConflictReport,
    ConflictResolution,
    ConflictResolutionError,
    apply_resolution,
    compute_content_sha256,
    detect_conflict,
)
from care.evolution_session import (
    EvolutionConfig,
    EvolutionMode,
    EvolutionPlan,
    EvolutionPlanError,
    EvolutionProgressTracker,
    GenerationStat,
    build_evolution_request,
    evolution_diff,
)
from care.help import (
    HelpRegistry,
    HelpRegistryExtension,
    KeyBinding,
    KeyCategory,
    TutorialStep,
    build_registry,
    default_registry,
    register_help_extension,
    unregister_help_extension,
)
from care.first_run import (
    FirstRunConfigError,
    FirstRunReport,
    ProbeResult,
    ProbeStatus,
    probe_mage,
    probe_memory,
    probe_platform,
    run_all_probes,
    write_initial_config,
)
from care.generation import (
    GenerationError,
    build_mage_config,
    build_mage_generator,
    run_generation,
)
from care.intermediate_artifacts import (
    IntermediateArtifact,
    IntermediateArtifactsView,
    project_intermediate_artifacts,
)
from care.mage_summary import (
    MetadataSummary,
    summarise_mage_result,
)
from care.marketplace import (
    MarketplaceError,
    MarketplaceListing,
    MarketplaceResult,
    search_marketplace,
)
from care.micro_evolution import (
    Evaluator,
    Individual,
    MicroEvolution,
    MicroEvolutionConfig,
    MicroEvolutionError,
    MicroEvolutionResult,
    Mutator,
    ObjectiveDirection,
    builtin_mutators,
    compute_pareto_front,
)
from care.catalog import (
    CapabilityCatalog,
    CapabilityCatalogEntry,
    EntryKind,
    build_catalog,
)
from care.preflight import (
    PreflightResult,
    validate_chain,
)
from care.profiling import (
    ProfilingSummary,
    StepProfile,
    project_profiling,
)
from care.replay import (
    ReplayError,
    ReplaySession,
    ReplayStep,
    load_replay,
)
from care.skills import (
    SkillPromotionError,
    promote_skill_to_memory,
)
from care.stage_regeneration import (
    RegenerateStage,
    StageArtifact,
    StageRegenerationError,
    regenerate_stage,
    supported_stages,
)
from care.tools import (
    LoadedTools,
    load_tools_into_context,
)

try:
    # Single source of truth lives in ``pyproject.toml`` (``version = ...``).
    # Read it from the installed distribution metadata so the literal is
    # never duplicated here.
    __version__ = _version("maestro-care")
except PackageNotFoundError:  # pragma: no cover - only when running uninstalled
    __version__ = "0.0.0"

__all__ = [
    "BulkImportEntry",
    "BulkImportReport",
    "CapabilityCatalog",
    "CapabilityCatalogEntry",
    "CapabilityPayload",
    "CapabilityPrimingError",
    "ChainExportError",
    "ConflictReport",
    "ConflictResolution",
    "ConflictResolutionError",
    "EntryKind",
    "EvolutionConfig",
    "EvolutionMode",
    "EvolutionPlan",
    "EvolutionPlanError",
    "EvolutionProgressTracker",
    "ExportFormat",
    "FirstRunConfigError",
    "FirstRunReport",
    "ExportResult",
    "GenerationError",
    "GenerationStat",
    "HelpRegistry",
    "HelpRegistryExtension",
    "Evaluator",
    "ImportStatus",
    "Individual",
    "IntermediateArtifact",
    "IntermediateArtifactsView",
    "KeyBinding",
    "KeyCategory",
    "LoadedTools",
    "MarketplaceError",
    "MarketplaceListing",
    "MarketplaceResult",
    "MetadataSummary",
    "MicroEvolution",
    "MicroEvolutionConfig",
    "MicroEvolutionError",
    "MicroEvolutionResult",
    "Mutator",
    "ObjectiveDirection",
    "PreflightResult",
    "ProbeResult",
    "ProbeStatus",
    "ProfilingSummary",
    "RegenerateStage",
    "ReplayError",
    "ReplaySession",
    "ReplayStep",
    "SkillPromotionError",
    "StageArtifact",
    "StageRegenerationError",
    "StepProfile",
    "TutorialStep",
    "__version__",
    "apply_resolution",
    "build_capability_payload",
    "build_catalog",
    "build_evolution_request",
    "build_mage_config",
    "build_mage_generator",
    "build_registry",
    "builtin_mutators",
    "compute_content_sha256",
    "compute_pareto_front",
    "default_registry",
    "detect_conflict",
    "evolution_diff",
    "export_chain",
    "import_chains",
    "load_replay",
    "load_tools_into_context",
    "project_intermediate_artifacts",
    "project_profiling",
    "probe_mage",
    "probe_memory",
    "probe_platform",
    "promote_skill_to_memory",
    "register_help_extension",
    "regenerate_stage",
    "run_all_probes",
    "run_generation",
    "search_marketplace",
    "summarise_mage_result",
    "supported_stages",
    "unregister_help_extension",
    "validate_chain",
    "write_initial_config",
]
