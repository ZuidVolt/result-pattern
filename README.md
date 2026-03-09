# Result Pattern

[![CI](https://github.com/ZuidVolt/result-pattern/actions/workflows/CI.yml/badge.svg)](https://github.com/ZuidVolt/result-pattern/actions/workflows/CI.yml)
[![Python Version](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/release/python-3140/)

A library designed to implement the **"Errors as Values"** pattern in Python 3.14+. It bridges the gap between pure functional safety and the pragmatic realities of the exception-heavy Python ecosystem.

---

## Installation

```sh
# Package name is 'result-pattern', but you import 'result'
$ pip install result-pattern
```

---

## Core Concept: Result (Sum Type)

A `Result` is a **Sum Type** representing a state of being either completely successful (`Ok`) or completely failed (`Err`).

### 1. Basic Usage
```python
from result import Ok, Err, Result, is_ok


def divide(a: int, b: int) -> Result[float, str]:
    if b == 0:
        return Err("Cannot divide by zero")
    return Ok(a / b)


res = divide(10, 2)
if is_ok(res):
    # res is narrowed to Ok[float]
    print(f"Success: {res.ok()}")
```

### 2. Modern Pattern Matching
Leverage Python 3.10+ native matching with optimized `__match_args__`:
```python
match divide(10, 0):
    case Ok(val):
        print(f"Value: {val}")
    case Err(msg):
        print(f"Error: {msg}")
```

### 3. Functional Chaining
Avoid nested `if` statements with linear pipelines:
```python
Ok(10).map(lambda x: x * 2).tap(print).and_then(lambda x: Ok(x + 1))
```

---

## Interoperability: The Catch System

Bridge the procedural world of exceptions to the functional world of Results.

### 1. `@catch` Decorator
Lift exception-throwing code into `Result` containers with zero boilerplate.
```python
from result import catch


@catch(ValueError)
def parse_int(s: str) -> int:
    return int(s)


parse_int("10")  # Ok(10)
parse_int("abc")  # Err(ValueError(...))
```

### 2. Exception Mapping (`map_to` & Enums)
Immediately sanitize wild exceptions into strictly typed Domain Enums.
```python
from enum import StrEnum
from result import catch


class ErrorCode(StrEnum):
    INVALID = "invalid_input"


# Single mapping
@catch(ValueError, map_to=ErrorCode.INVALID)
def risky_op(s: str): ...


# Multiple mapping dictionary
@catch({ValueError: ErrorCode.INVALID, KeyError: "missing"})
def complex_op(x): ...
```

### 3. `catch_call` Inline
Execute standard library functions inline without opening context blocks.
```python
from result import catch_call
import json

res = catch_call(json.JSONDecodeError, json.loads, '{"key": "value"}')
```

### 4. `catch` Context Manager
Lasso an entire block of wild Python code and capture the result safely.
```python
from result import catch
import json

with catch(json.JSONDecodeError) as safe_block:
    data = json.loads(payload)
    # Perform complex logic...
    safe_block.set(data["nested"]["key"])

# result is narrowed to Result[T, JSONDecodeError]
print(safe_block.result)
```

---

## Monadic Orchestration: Do-Notation

Write procedural-looking code that automatically handles short-circuiting logic.

```python
from result import do_notation, Do


@do_notation
def compile_pipeline(source: str) -> Do[str, Exception]:
    tokens = yield tokenize(source)  # Returns list[Token] or short-circuits Err
    ast = yield parse(tokens)  # Returns AST or short-circuits Err
    code = yield generate(ast)
    return code  # Automatically wrapped in Ok
```

---

## Combinators: Advanced Utilities

| Utility | Description |
| :--- | :--- |
| `validate()` | **Applicative**: Accumulates all errors instead of failing fast. |
| `traverse()` | Maps a fallible function over an iterable, failing fast. |
| `flow()` | A sequential pipeline macro for piped data transformations. |
| `succeeds()` | Filters a collection of Results, returning only the success values. |
| `partition_exceptions()` | Splits a mixed list of `[Value, Exception]` into `[Ok, Err]`. |

---

## Outcome: Product Type (Go/Odin Style)

While `Result` is mathematically pure, real-world systems often require **fault tolerance**. `Outcome[T, E]` is a **Product Type** that holds both a partial success value and diagnostic baggage simultaneously.

### Advantages:
* **Native Unpacking**: Inherits from `NamedTuple` for `val, err = do_work()` syntax.
* **Fault Tolerance**: Retain your AST or data payload even if warnings/errors occurred.
* **Odin Style**: Use `.to_result()` inside a `@do_notation` block to simulate Odin's `or_return`.

### Usage:
```python
from result import Outcome


def parse_with_diagnostics(source: str) -> Outcome[AST, list[str]]:
    # build AST and accumulate errors...
    return Outcome(ast, accumulated_errors)


# 1. Procedural Unpacking (Go/Odin style)
ast, errors = parse_with_diagnostics(src)
if errors:
    print(f"Warnings: {errors}")

# 2. Accumulation
new_outcome = Outcome(node, "e1").push_err("e2").merge(other_outcome)


# 3. Transition to Strict (or_return)
@do_notation
def strict_flow():
    # .to_result() halts execution if any errors exist
    ast = yield parse_with_diagnostics(src).to_result()
    return emit(ast)
```

---

## API Philosophy

### Zero-Escape Safety
Panics are isolated in the `.unsafe` namespace. Direct `.unwrap()` access is disabled at the root level to encourage safe functional patterns.
```python
res = Err("fail")
res.unsafe.unwrap()  # Explicit panic
```

### True Covariance
Full support for inheritance subtyping (e.g., `Ok[Dog]` is assignable to `Result[Animal, Any]`).
