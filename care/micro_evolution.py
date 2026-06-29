"""Local micro-evolution fallback (TODO §7 P2).

When the GigaEvo Platform isn't reachable — offline demos, CI
smoke tests, single-machine experiments — CARE can still run a
**tiny in-process GA** over a seed chain. This module owns the
algorithm; the user supplies a mutator and an evaluator and gets
back a structured result describing the best individual and the
per-generation history.

Design intentionally narrow:

* **No threading / no asyncio.** The Platform is the production
  evolution path; this is a fallback for demos. Each generation
  runs sequentially so the loop is trivially debuggable.
* **Caller-provided mutator + evaluator.** The library doesn't
  prescribe chain mutation strategies — CARL chains have eight
  step types with different semantics, and CARE shouldn't bake
  in opinions about which slot to perturb. We do ship a handful
  of safe built-in mutators in :func:`builtin_mutators` for
  demos that don't want to write their own.
* **Deterministic with a seed.** Tests and replays pass
  ``MicroEvolutionConfig.seed`` to pin RNG state — same seed
  + same evaluator → identical result. Without a seed the
  algorithm uses ``random.Random()``'s default entropy.
* **Elitism + tournament selection.** The top-K elites carry
  over to the next generation unchanged; the rest are filled
  by mutating winners of small random tournaments. Simple
  enough to fit in 200 lines, expressive enough for the
  three-quarter demo loop.

This is **not** a CARL-runtime integrator — the
TODO bullet calls it out as using CARL's executor as the fitness
evaluator, but wiring an actual ``ReasoningChain.execute_async``
into the evaluator is a separate task that depends on the runtime
work in §5. The interface here is shape-compatible: the user can
pass a function that wraps ``execute_async`` + a custom scoring
heuristic and the algorithm doesn't care.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Protocol

Mutator = Callable[[dict[str, Any], random.Random], dict[str, Any]]
"""Function that takes a chain dict + RNG and returns a mutated
chain dict. Must not mutate the input — return a fresh dict so the
elite path can preserve the original."""


class Evaluator(Protocol):
    """Scoring contract for one individual.

    A scalar return is treated as a single fitness dimension —
    higher is better. A dict return supports multi-objective
    scoring; ``MicroEvolution`` reduces it to a scalar by
    summing the values (callers wanting Pareto-style selection
    can wrap their dict evaluator + override
    :class:`MicroEvolutionConfig.score_reducer`).
    """

    def __call__(self, chain: dict[str, Any]) -> float | dict[str, float]:
        ...


@dataclass(frozen=True)
class Individual:
    """One member of the population.

    Frozen so the result snapshot can be passed around without
    defensive copies. ``chain`` is the chain dict — callers
    inspecting it should not mutate (frozen guards the wrapper
    but not the nested dict, so this is a convention).

    Fields:
        chain: Chain dict the evaluator scored.
        score: Scalar fitness (reduced from the evaluator's
            return). Higher is better.
        breakdown: When the evaluator returned a dict, the raw
            per-objective scores are preserved here for the
            run's history. Empty when the evaluator returned a
            scalar.
        generation: 0-indexed generation the individual was
            born into.
    """

    chain: dict[str, Any]
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    generation: int = 0


@dataclass(frozen=True)
class MicroEvolutionConfig:
    """Knobs for :class:`MicroEvolution`.

    Defaults tuned for fast demo runs (5×4 = 20 evaluations).
    """

    population_size: int = 5
    generations: int = 4
    elitism: int = 1
    """How many top individuals carry over unchanged per
    generation. Must be ≤ ``population_size``."""
    tournament_size: int = 3
    """Number of contenders per tournament selection. Each
    non-elite slot in the next generation pulls one winner
    from a fresh tournament of this size."""
    seed: int | None = None
    """RNG seed for reproducibility. ``None`` uses default
    entropy."""
    score_reducer: Callable[[dict[str, float]], float] = staticmethod(
        lambda d: sum(d.values())
    )
    """How to collapse a multi-objective evaluator output into a
    scalar fitness. Default sums every value — works fine when
    all objectives are aligned ("higher is better"). Pass a
    custom reducer for weighted / Pareto behaviour."""


ObjectiveDirection = Literal["maximize", "minimize"]
"""How a single objective is interpreted for Pareto comparison.

:class:`Individual.breakdown` stores raw evaluator values; the
direction tells :func:`compute_pareto_front` whether bigger or
smaller is "better" for each key. Defaults to ``"maximize"`` for
every key when callers don't supply directions — same convention
as the scalar :class:`Evaluator` contract ("higher is better")."""


@dataclass(frozen=True)
class MicroEvolutionResult:
    """What :meth:`MicroEvolution.run` returned.

    Frozen so the result can be logged / persisted as-is.
    """

    best: Individual
    history: tuple[Individual, ...]
    """One entry per generation — the best individual of that
    generation. Length equals ``config.generations``."""
    population: tuple[Individual, ...]
    """The final generation, sorted by score descending."""
    evaluations: int
    """Total evaluator calls. Equals
    ``population_size + (generations - 1) * (population_size -
    elitism)`` when the loop runs to completion (elites are
    cached, not re-scored)."""

    def pareto_front(
        self,
        directions: dict[str, ObjectiveDirection] | None = None,
    ) -> tuple[Individual, ...]:
        """Return the non-dominated subset of ``self.population``.

        Convenience wrapper around :func:`compute_pareto_front` so
        the result object exposes the Pareto front directly —
        callers don't have to know which collection to feed in.

        Args:
            directions: Per-objective ``"maximize"`` / ``"minimize"``
                mapping. Keys are the objective names that appear
                in :class:`Individual.breakdown`. Missing keys
                default to ``"maximize"``. ``None`` treats every
                objective as maximise.

        Returns:
            Tuple of non-dominated individuals from the final
            generation. When the evaluator returned scalars (no
            ``breakdown``), this falls back to a single-objective
            comparison on :attr:`Individual.score`.
        """
        return compute_pareto_front(
            self.population,
            directions=directions,
        )


class MicroEvolutionError(RuntimeError):
    """Raised on misconfiguration — invalid population_size /
    elitism / tournament_size."""


# ---------------------------------------------------------------------------
# Pareto-front computation (TODO §7 P2 multi-objective)
# ---------------------------------------------------------------------------


def compute_pareto_front(
    individuals: Iterable[Individual],
    *,
    directions: dict[str, ObjectiveDirection] | None = None,
) -> tuple[Individual, ...]:
    """Return the non-dominated subset of ``individuals``.

    An individual `a` *dominates* `b` when `a` is at least as
    good as `b` on every objective and strictly better on at
    least one. The Pareto front is the set of individuals nobody
    else dominates — the "tradeoff frontier" the user picks from
    in the future EvolutionScreen.

    Args:
        individuals: Iterable of :class:`Individual`. Order is
            preserved on the returned tuple so callers can keep
            insertion-order semantics; ``pareto_front()[0]`` is
            the first non-dominated individual in input order,
            not a sort by score.
        directions: Per-objective ``"maximize"`` / ``"minimize"``
            mapping. Missing keys default to ``"maximize"``;
            ``None`` treats every key as maximise.
            Latency / cost objectives (where lower is better)
            should set ``"minimize"`` here.

    Returns:
        Tuple of non-dominated individuals. Empty input yields
        an empty tuple. When an individual has an empty
        ``breakdown``, the function falls back to scalar
        :attr:`Individual.score` comparison (always "maximise").
        Missing objectives on an individual are treated as the
        worst possible value for that direction (matches
        Platform §4.2's "missing scores as worst" convention).
    """
    pool: list[Individual] = list(individuals)
    if not pool:
        return ()
    directions = dict(directions or {})

    # Collect every objective key any individual has.
    all_keys: list[str] = []
    seen: set[str] = set()
    use_scalar_fallback = True
    for ind in pool:
        if ind.breakdown:
            use_scalar_fallback = False
            for k in ind.breakdown:
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)

    if use_scalar_fallback:
        # No breakdown anywhere — single-objective on `.score`.
        # Direction is implicitly "maximise" because
        # `Individual.score` is documented as "higher is better".
        return tuple(_pareto_scalar(pool))

    front: list[Individual] = []
    for candidate in pool:
        dominated = False
        for other in pool:
            if other is candidate:
                continue
            if _dominates(other, candidate, all_keys, directions):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return tuple(front)


def _objective_value(
    ind: Individual,
    key: str,
    direction: ObjectiveDirection,
) -> float:
    """Read ``ind.breakdown[key]``; fall back to the worst possible
    value for the given direction when missing."""
    val = ind.breakdown.get(key)
    if val is None:
        return -math.inf if direction == "maximize" else math.inf
    return float(val)


def _dominates(
    a: Individual,
    b: Individual,
    keys: list[str],
    directions: dict[str, ObjectiveDirection],
) -> bool:
    """True when ``a`` weakly beats ``b`` on every objective and
    strictly beats it on at least one."""
    strictly_better_somewhere = False
    for key in keys:
        direction = directions.get(key, "maximize")
        av = _objective_value(a, key, direction)
        bv = _objective_value(b, key, direction)
        if direction == "maximize":
            if av < bv:
                return False
            if av > bv:
                strictly_better_somewhere = True
        else:  # minimize
            if av > bv:
                return False
            if av < bv:
                strictly_better_somewhere = True
    return strictly_better_somewhere


def _pareto_scalar(pool: list[Individual]) -> list[Individual]:
    """Single-objective fallback when every individual has empty
    ``breakdown`` — pick the max-score set (ties all included)."""
    best_score = max(ind.score for ind in pool)
    return [ind for ind in pool if ind.score == best_score]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class MicroEvolution:
    """Runs the GA loop.

    Construct with the seed chain + mutator + evaluator, then
    call :meth:`run`. See module docstring for design notes.
    """

    def __init__(
        self,
        seed_chain: dict[str, Any],
        evaluator: Evaluator,
        *,
        mutator: Mutator,
        config: MicroEvolutionConfig | None = None,
    ) -> None:
        if not isinstance(seed_chain, dict):
            raise MicroEvolutionError(
                f"seed_chain must be a dict; got {type(seed_chain).__name__}"
            )
        self._seed = seed_chain
        self._evaluator = evaluator
        self._mutator = mutator
        self._config = config or MicroEvolutionConfig()
        self._validate_config()
        self._rng = random.Random(self._config.seed)

    def run(self) -> MicroEvolutionResult:
        """Execute the GA loop and return a populated
        :class:`MicroEvolutionResult`."""
        cfg = self._config

        # Generation 0: seed + mutated variants.
        population: list[Individual] = [self._score(self._seed, generation=0)]
        for _ in range(cfg.population_size - 1):
            child = self._mutator(copy.deepcopy(self._seed), self._rng)
            population.append(self._score(child, generation=0))
        population.sort(key=lambda i: i.score, reverse=True)

        history: list[Individual] = [population[0]]
        evaluations = len(population)

        for gen in range(1, cfg.generations):
            # Carry over elites verbatim — their score is cached.
            elites = population[: cfg.elitism]
            new_pop: list[Individual] = list(elites)

            while len(new_pop) < cfg.population_size:
                winner = self._tournament(population)
                child_chain = self._mutator(
                    copy.deepcopy(winner.chain), self._rng
                )
                new_pop.append(self._score(child_chain, generation=gen))
                evaluations += 1

            new_pop.sort(key=lambda i: i.score, reverse=True)
            population = new_pop
            history.append(population[0])

        return MicroEvolutionResult(
            best=population[0],
            history=tuple(history),
            population=tuple(population),
            evaluations=evaluations,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _score(
        self, chain: dict[str, Any], *, generation: int
    ) -> Individual:
        raw = self._evaluator(chain)
        if isinstance(raw, dict):
            breakdown = {str(k): float(v) for k, v in raw.items()}
            score = float(self._config.score_reducer(breakdown))
        else:
            breakdown = {}
            score = float(raw)
        return Individual(
            chain=chain,
            score=score,
            breakdown=breakdown,
            generation=generation,
        )

    def _tournament(self, population: list[Individual]) -> Individual:
        """Pick the highest-scoring individual from a random sample."""
        size = min(self._config.tournament_size, len(population))
        contenders = self._rng.sample(population, k=size)
        return max(contenders, key=lambda i: i.score)

    def _validate_config(self) -> None:
        cfg = self._config
        if cfg.population_size < 1:
            raise MicroEvolutionError(
                f"population_size must be >= 1; got {cfg.population_size}"
            )
        if cfg.generations < 1:
            raise MicroEvolutionError(
                f"generations must be >= 1; got {cfg.generations}"
            )
        if cfg.elitism < 0 or cfg.elitism > cfg.population_size:
            raise MicroEvolutionError(
                f"elitism must be in [0, {cfg.population_size}]; "
                f"got {cfg.elitism}"
            )
        if cfg.tournament_size < 1:
            raise MicroEvolutionError(
                f"tournament_size must be >= 1; got {cfg.tournament_size}"
            )


# ---------------------------------------------------------------------------
# Built-in mutators (safe defaults for demos)
# ---------------------------------------------------------------------------


def noop_mutator(chain: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Returns the chain unchanged. Useful as a placeholder when
    the caller wants to exercise the evaluator + selection logic
    without actually mutating anything."""
    return chain


def perturb_step_titles(
    chain: dict[str, Any], rng: random.Random
) -> dict[str, Any]:
    """Tack a small random suffix onto one step's `title`.

    Doesn't change the chain's semantic shape — just gives the
    evaluator a way to tell variants apart by inspecting the
    title field. Handy for tests where the "fitness" is a
    function of the title length / hash.
    """
    steps = chain.get("steps")
    if not isinstance(steps, list) or not steps:
        return chain
    idx = rng.randrange(len(steps))
    step = steps[idx]
    if isinstance(step, dict) and "title" in step:
        step["title"] = f"{step['title']}_{rng.randint(0, 9999)}"
    return chain


def drop_optional_step(
    chain: dict[str, Any], rng: random.Random
) -> dict[str, Any]:
    """Drop a non-first step that nothing else depends on.

    Conservative: only drops a step when removing it can't leave
    a dangling ``dependencies`` reference. The seed chain stays
    intact when no such step exists.
    """
    steps = chain.get("steps")
    if not isinstance(steps, list) or len(steps) <= 1:
        return chain
    # Build dependency closure.
    dependents: dict[int, set[int]] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        for dep in step.get("dependencies", []) or []:
            dependents.setdefault(int(dep), set()).add(int(step.get("number", -1)))
    candidates: list[int] = []
    for i, step in enumerate(steps):
        if i == 0 or not isinstance(step, dict):
            continue
        number = step.get("number")
        if isinstance(number, int) and not dependents.get(number):
            candidates.append(i)
    if not candidates:
        return chain
    chain["steps"] = [
        s for i, s in enumerate(steps) if i != rng.choice(candidates)
    ]
    return chain


def compose_mutators(*mutators: Mutator) -> Mutator:
    """Pick one of the supplied mutators uniformly at random per call."""
    if not mutators:
        return noop_mutator

    def composed(chain: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        return rng.choice(mutators)(chain, rng)

    return composed


def builtin_mutators() -> Mutator:
    """Bundle the safe defaults into one composed mutator —
    handy when callers don't want to think about which mutator
    to pick."""
    return compose_mutators(perturb_step_titles, drop_optional_step)


__all__ = [
    "Evaluator",
    "Individual",
    "MicroEvolution",
    "MicroEvolutionConfig",
    "MicroEvolutionError",
    "MicroEvolutionResult",
    "Mutator",
    "ObjectiveDirection",
    "builtin_mutators",
    "compose_mutators",
    "compute_pareto_front",
    "drop_optional_step",
    "noop_mutator",
    "perturb_step_titles",
]
