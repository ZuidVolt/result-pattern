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
- **Development Workflow:** Every logical change **must** be verified by running `just check && just test`.

## Architecture & Logic

- **Functional Primitive (`result.py`)**: The core module housing the `Result` type system (`Ok`, `Err`) and functional utilities.
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
- **Absolute Preservation:** All surgical edits (`replace`, `write_file`) **must** preserve existing docstrings, examples, and technical notes with 100% fidelity. 
- **Zero Truncation:** Never omit or 'summarize' documentation blocks during a code change. 
- **Verification:** Before applying a `replace` call, you **must** use `read_file` to verify the exact string content of the surrounding documentation to ensure no data is lost in the transaction.
