# Result Pattern

A lightweight, single-file library designed to implement the 'Errors as Values' pattern in Python 3.14+. This library bridges the gap between pure functional safety and the pragmatic realities of the exception-heavy Python ecosystem.

# Project Overview

## 1. Mandatory Tooling & Environment
- **Python 3.14+:** Usage of Python 3.14 is **mandatory**. 
- **Modern Idioms:** 
    - Leverage PEP 649 (Native Lazy Type Evaluation). Do **not** use string-wrapped return types or forward references.
    - **Exception Syntax:** Use unparenthesized multiple exceptions (PEP 758) for all handlers (e.g., `except ValueError, TypeError:`). Parentheses are only used if capturing with `as`.
    - **Finally Blocks:** Never use `return`, `break`, or `continue` inside `finally` blocks (PEP 765).
- **Package Manager:** `uv` is the exclusive tool for dependency management and environment execution.
## Development Workflow

- **Modern Idioms**: Use Python 3.14+ features (PEP 649, 758, 765) exclusively.
- **Verification**: Every change must be verified by running `just check && just test`.
- **Testing Philosophy**: Prioritize high-signal testing strategies through a **Dual-Layered Suite**:
    *   **Public API Layer (`tests/test_result_api.py`)**: 
        *   **Zero-Tolerance Type Safety**: This suite must pass all type checks (`mypy`, `ty`, `basedpyright`) with **zero** suppression comments (`type: ignore`, `pyright: ignore`, `cast`). 
        *   **Laws & Integration**: Verifies Algebraic Laws (Functor/Monad), Style Equivalence between `@do_notation` and chaining, and complex Data Pipelines. This ensures a perfect, type-safe developer experience for consumers.
    *   **Gradual Typing & Inference Layer (`tests/test_result_api_inference.py`)**:
        *   **Ergonomic Validation**: Specifically tests how the library performs under gradual typing where users provide minimal or no annotations.
        *   **Inference Sections**: Divided into "Zero Annotation" (testing pure inference) and "Basic Annotation" (testing standard function-level hints) to ensure type info is correctly traced through the system without erasure.
        *   **Pragmatic Suppressions**: This file disables strict-mode pedantry (like `basedpyright`'s unknown variable warnings) at the file level to focus on how `mypy` and `ty` actually assist real-world developers.
    *   **Internal Implementation Layer (`tests/test_result_internal.py`)**: 
        *   **Behavioral Focus**: Houses educational safeguards, unreachable coverage mop-ups, and tests for private API invariants.
        *   **Intentional Type Bypasses**: This file uses file-level ignores to focus exclusively on runtime correctness and edge-case coverage (e.g., testing "panics" or intentional API misuse) without type-system noise.
- **Documentation Fidelity**: Docstrings are functional code; preserve them with 100% accuracy during edits.

## Architecture & Logic

- **Functional Primitive (`result.py`)**: The core module housing the `Result` type system (`Ok`, `Err`) and functional utilities.
- **Dual Generic System (Covariance)**: To ensure true covariance (supporting inheritance like `Dog` -> `Animal`), the library uses a hybrid typing approach:
    - **Modern PEP 695**: Used for type aliases (`type Result[T, E]`) and standalone utilities for clean, readable API signatures.
    - **Legacy Generic/TypeVar**: Used for class definitions (`class Ok(Generic[T_co])`) with explicit `covariant=True` parameters. This is necessary because automated variance inference in PEP 695 is often too conservative for functional methods like `.map()` or `.tap()`.
- **Zero-Escape Safety**: Isolate crashing operations (panics) within the `.unsafe` namespace to ensure that 'unwrapping' is always an intentional, visible choice.
- **Pragmatic Interop**: Provides 'lifting' tools like `@safe` to seamlessly convert standard Python exception-throwing code into functional containers.

## Functional Orchestration Guidelines

To maintain a high standard of readability, use the following rules when orchestrating logic:

- **Linear Chaining (`.and_then`)**: Preferred for simple, linear pipelines where state does not need to be accumulated.
- **Do-Notation (`@do_notation`)**: Preferred for complex logic with multiple variables and branching. This uses Python generators to simulate monad-like behavior:
    - `val = yield result`: Unwraps `Ok(val)` or early-returns `Err`.
    - `return val`: Automatically wraps the final value in an `Ok(val)`.
- **Exhaustive Matching**: Use `match` statements or the `.match()` method to handle both success and failure states explicitly.

## Coding Style & Patterns

This library exhibits a modern and disciplined Python coding style with influences from functional programming and Rust:

- **Result Type API**: A robust, Rust-inspired functional primitive for error handling:
    - `.map(func)`: Transforms the success value.
    - `.map_err(func)`: Transforms the error value.
    - `.and_then(func)`: Chains operations that return a `Result` (flatMap).
    - `.unsafe.unwrap()`: Force-extracts the success value (explicit panic).
    - `.unwrap_or(default)`: Safe extraction with a fallback.
- **Type Narrowing**: Standalone `is_ok(result)` and `is_err(result)` functions using PEP 742 `TypeIs` for precise narrowing in conditional blocks.
- **Complexity Management**: Strictly adheres to complexity limits (Ruff `C901`) by extracting focused sub-helpers.
- **Type Safety**: Leverages Python 3.14+ features like PEP 695 type aliases and generic syntax for classes and functions, ensuring robust static analysis with Basedpyright.
- **Zero Suppressions**: The codebase contains zero `# noqa` comments (outside of generated code or unavoidable edge cases), maintaining a high bar for linter compliance and maintainability.
- **Modern Tooling**: Managed via `uv` for dependency management and `just` for task automation.

## Documentation & Meta-Data Integrity

The project treats inline documentation (docstrings) as **functional code**. 
- **Absolute Preservation:** All surgical edits (`replace`) **must** preserve existing docstrings, examples, and technical notes with 100% fidelity. 
- **Tooling Ban:** **NEVER** use `write_file` to modify existing source code files as it can destroy structure and documentation. You **must** use the `replace` tool for all surgical edits. `write_file` is only permitted for creating entirely new files.
- **Zero Truncation:** Never omit or 'summarize' documentation blocks during a code change. 
- **Verification:** Before applying a `replace` call, you **must** use `read_file` to verify the exact string content of the surrounding documentation to ensure no data is lost in the transaction.
