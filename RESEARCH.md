# Research & Roadmap: "Errors as Values" Expansion

This document synthesizes research from Gleam, Rust, OCaml, and F# to define the next evolution of the `result-pattern` library.

---

## 1. Cross-Language Syntax Comparison

| Feature | Rust | Gleam | F# | OCaml | Our Python Implementation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Monadic Bind** | `?` operator | `use` keyword | `let!` (CE) | `let*` operator | `@do_notation` (Generators) |
| **Interop** | `transpose` | `result.nil_error` | `Option.toResult` | `Result.to_option` | `.ok()` / `.err()` / `from_optional()` |
| **Bulk Merging** | `collect()` | `result.all` | `map2` / `map3` | `Result.iter` | `combine()` / `partition()` |
| **Lifting** | `Try` trait | `result.try` | `safe` patterns | Standard `try/with` | `@safe` decorator |

---

## 2. Strategic Roadmap: "The Ultimate Result"

Based on the synthesis of functional patterns and Pythonic ergonomics, the following features are planned:

### Phase 1: The Interop Layer (Highest Priority)
*   **`transpose()`**: Allows swapping between `Result[T | None, E]` and `Result[T, E] | None`. This is critical for database operations where a "Not Found" state is represented by `None`.
*   **`safe` Context Manager**: Extends the `@safe` decorator to work as a context manager for localized exception trapping.
    ```python
    with safe(ValueError) as res:
        x = int(input)
    # res is now Ok(x) or Err(ValueError)
    ```

### Phase 2: Ergonomic Orchestration
*   **`product()` (Zip)**: Merges two results into a tuple.
    `Ok(1).product(Ok(2)) -> Ok((1, 2))`
*   **`map2` / `map3`**: Zips results and applies a function in one step. Inspired by F#.
*   **Implicit Coercion**: Adds a `remap` parameter to `@do_notation` to automatically convert low-level errors (e.g., `DBError`) into high-level domain errors (e.g., `AppError`) during yielding.

### Phase 3: Advanced Recovery
*   **`any()`**: Returns the first `Ok` variant from a collection, or a list of all `Err`s if none succeeded.
*   **`unwrap_or_default()`**: Provides a shortcut for `.unwrap_or(type())` for standard types like `int`, `str`, and `list`.

---

## 3. Implementation Philosophy

1.  **Zero-Dependency**: All expansions must remain within the Python 3.14 standard library.
2.  **Type-First**: No feature will be added if it cannot be represented with 100% type-accuracy in `basedpyright`.
3.  **Educational Guardrails**: Continue using `__getattr__` to intercept legacy or unsafe patterns and guide developers toward best practices.
