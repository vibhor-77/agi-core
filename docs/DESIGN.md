# Architecture Design Document

## 1. The Idea

The system implements a general learning algorithm built on four pillars:
**Composition**, **Decomposition**, **Memory**, and **Drive Signal**.

Starting from minimal seeds (simple primitives), the algorithm compounds:
composing primitives into programs, decomposing complex problems into simpler
ones, remembering what works (library), driven by a quality signal (energy).
Each round of wake-sleep learning builds higher-level abstractions from
lower-level ones.

The same algorithm works for any domain. Only the sensors, actuators, and
data differ. ARC grids, list operations, text adventures — the core loop
is invariant.

**The core claim**: Given the right primitives and enough wake-sleep rounds,
the library grows and harder problems become tractable without deeper search.
If the compounding curve bends upward, the framework is working.


## 2. The Four Pillars as Code

| Pillar | Interface | Key Methods |
|--------|-----------|-------------|
| **Composition** | `Grammar` | `compose()`, `base_primitives()`, `inject_library()` |
| **Decomposition** | `Grammar` | `decompose()`, `recompose()` |
| **Memory** | `Memory` | `get_library()`, `add_to_library()`, `store_solution()` |
| **Drive Signal** | `DriveSignal` | `prediction_error()`, `complexity_cost()`, `energy()` |

The `Environment` interface bridges the agent to the world: `execute()` runs
programs, `load_task()` provides observations, `domain_wake_phases()` supplies
domain-specific search strategies.

**Energy = error + complexity** (MDL / Occam's Razor):
```
E(candidate) = alpha * prediction_error + beta * complexity_cost
```
Lower energy = better program. Beta penalizes complexity, preventing
overfitting to training examples.


## 3. Layer Diagram

```
                    +--------------------------+
                    |          CLI             |
                    |  python -m common ...    |
                    +-----------+--------------+
                                |
                    +-----------+--------------+
                    |     common/benchmark     |
                    |  pipeline, presets,      |
                    |  progress tracking       |
                    +-----------+--------------+
                                |
              +-----------------+-----------------+
              |                                   |
    +---------+----------+             +----------+---------+
    |      core/         |             |    domains/<name>/ |
    |  learner.py        |<--- plugs --+  environment.py    |
    |  interfaces.py     |    into     |  grammar.py        |
    |  types.py          |             |  drive.py          |
    |  config.py         |             |  phases.py         |
    |  memory.py         |             |  adapter.py        |
    |  results.py        |             |  primitives.py     |
    +--------------------+             +--------------------+

    INVARIANT: core/ never imports anything from domains/
```

- **core/** — The domain-agnostic algorithm. Types, interfaces, learner,
  sleep, curriculum, exhaustive enumeration.
- **domains/** — Pluggable domain implementations. Each provides a
  `DomainAdapter` plus implementations of Environment, Grammar, DriveSignal.
  Domain-specific wake phases live here too.
- **common/** — Benchmark runner, pipeline orchestration, metrics, CLI.


## 4. Core Types

Defined in `core/types.py`:

- **`Primitive`** — An atomic operation. Three kinds:
  - `transform`: Grid -> Grid (or value -> value)
  - `perception`: Grid -> scalar value (feature extraction)
  - `parameterized`: values -> Grid -> Grid factory (e.g., replace_color(from, to))
- **`Program`** — A tree of primitives. `root` is the outermost primitive,
  `children` are sub-programs. Evaluation is recursive.
- **`Task`** — A problem: list of (input, output) training examples plus
  optional test inputs/outputs for validation.
- **`ScoredProgram`** — A Program with its energy, prediction error, complexity
  cost, max example error, and example solve score.
- **`LibraryEntry`** — A learned abstraction: a Program fragment extracted by
  sleep, with usefulness score and provenance.


## 5. The Compounding Cycle

```
    +-------+                    +--------+
    | WAKE  |  --- solutions --> | SLEEP  |
    | search|  --- attempts -->  | extract|
    +---+---+                    +---+----+
        ^                            |
        |                            v
        +--- library entries as --+--+
             new primitives
```

**Wake**: For each task, enumerate programs up to depth 3, then run
domain-specific phases (object decomposition, cross-reference, correction, etc.).
Score each candidate by energy. Store solutions and best attempts.

**Sleep**: Extract recurring sub-programs from solutions and near-misses.
Score by cross-task transfer (appears in multiple tasks = more useful).
Add high-scoring fragments to the library. Decay old entries.

**Compounding**: In the next wake round, library entries become new primitives.
A depth-1 program using a library entry is effectively depth-2+ from the
original primitives. This enables deeper compositions without deeper search.
This IS the compounding mechanism.


## 6. How Domains Plug In

Each domain implements four interfaces plus an adapter:

1. **`Environment`** — Execute programs, load tasks, register primitives.
   Override `domain_wake_phases()` to return domain-specific search strategies.
2. **`Grammar`** — Define base primitives, composition rules, mutation/crossover.
   Optionally provide predicates, essential pair concepts, task priorities.
3. **`DriveSignal`** — Define prediction error metric and complexity cost.
4. **`Memory`** — Usually just use `InMemoryStore` from core.
5. **`DomainAdapter`** — Wire it all together: name, interface construction,
   task loading, config defaults, post-run hooks.

Example: ARC provides 48 atomic primitives, 12 predicates, 9 domain-specific
wake phases (object decomposition, cross-reference, local rules, procedural
learning, synthesis, conditional search, color fix, input-prediction correction).

The core never imports domain code. Domains register themselves via the adapter
pattern, discovered at runtime.


## 7. Data Flow

```
CLI args
  |
  v
common/benchmark.py
  |-- loads DomainAdapter by name
  |-- adapter.create_interfaces() -> (env, grammar, drive)
  |-- adapter.load_tasks() -> [Task, ...]
  |-- creates Learner(env, grammar, drive, memory)
  |
  v
Learner.run_curriculum(tasks, config)
  |
  +-- for each round:
  |     |
  |     +-- WAKE: for each task (parallel):
  |     |     |-- exhaustive enumeration (depth 1-3)
  |     |     |-- domain_wake_phases (from env)
  |     |     +-- store solution or best attempt
  |     |
  |     +-- SLEEP: extract subtrees, score, add to library
  |     |
  |     +-- report: solve rates, library growth
  |
  v
Results + culture file (library + solutions)
```


## 8. Key Design Decisions

- **No neural networks.** All search is exhaustive + compositional. Programs
  are interpretable trees of named primitives. Deterministic, reproducible.

- **Exhaustive over beam search.** Depth-3 enumeration with smart pool
  pruning finds more solutions than beam search with random mutations.
  The search is bounded by eval budget, not time.

- **Minimal seeds.** Start with truly atomic primitives (rotate, flip, crop,
  fill, recolor). Complex behaviors emerge from composition, not from
  hand-coding complex primitives.

- **Domain phases via interface, not hardcoding.** The core provides only
  exhaustive enumeration. All structural strategies (per-object, cross-reference,
  correction, etc.) are domain-provided wake phases. This keeps the core
  clean and allows each domain to define its own search strategies.

- **LOOCV validation.** Learned rules (local rules, color maps, corrections)
  are validated via leave-one-out cross-validation to prevent overfitting to
  the small number of training examples (typically 2-4 in ARC).

- **Energy = MDL.** The drive signal operationalizes Occam's Razor. Among
  programs that match training examples, prefer simpler ones. Beta controls
  the complexity penalty.


## 9. Glossary

| Term | Definition |
|------|-----------|
| **Wake** | Search phase: enumerate and evaluate candidate programs |
| **Sleep** | Consolidation phase: extract library entries from solutions |
| **Library** | Collection of learned program fragments (abstractions) |
| **Primitive** | Atomic operation (transform, perception, or parameterized) |
| **Program** | Tree of primitives, evaluated recursively |
| **Energy** | Scoring function: error + complexity penalty |
| **Pareto front** | Best programs at each complexity level |
| **Culture** | Saved state: library + solutions, persisted to disk |
| **Compounding** | Library entries as primitives enable deeper compositions |
| **Domain adapter** | Plugin interface connecting a domain to the core |
| **WakeContext** | Shared state passed through all wake phases |
| **LOOCV** | Leave-one-out cross-validation for rule learning |
| **MDL** | Minimum description length — Occam's Razor formalized |
