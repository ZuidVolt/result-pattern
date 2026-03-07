# Result Pattern

[![CI](https://github.com/ZuidVolt/result-pattern/actions/workflows/CI.yml/badge.svg)](https://github.com/ZuidVolt/result-pattern/actions/workflows/CI.yml)
[![Python Version](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/release/python-3140/)

zero-dependency Result type for Modern Python (3.14+). This library implements the "Errors as Values" pattern with a focus on strict type safety, safety guardrails, and ergonomic functional orchestration.

## Installation

```sh
# The package is 'result-pattern', but you import 'result'
$ pip install result-pattern
```

## Summary

The core idea is that a result value can be either `Ok(value)` or `Err(error)`. Unlike standard Python where errors are implicit side-effects (Exceptions), `Result[T, E]` makes failure modes a first-class part of your API.

### Transforming Code

**Legacy Python (Exceptions/Tuples):**
```python
def get_user(user_id: int) -> User | None:
    if not database_active():
        raise ConnectionError("DB Down")
    return db.query(user_id)

try:
    user = get_user(1)
    if user:
        print(user.name)
except ConnectionError as e:
    print(f"Failed: {e}")
```

**Modern Python (Result Pattern):**
```python
from result import Ok, Err, Result, is_ok

def get_user(user_id: int) -> Result[User, str]:
    if not database_active():
        return Err("DB Down")
    user = db.query(user_id)
    return Ok(user) if user else Err("Not Found")

res = get_user(1)
if is_ok(res):
    # res is narrowed to Ok[User]
    print(res.ok().name)
else:
    # res is narrowed to Err[str]
    print(f"Failed: {res.err()}")
```

### Pattern Matching (Python 3.10+)

Thanks to aligned `__match_args__`, you can use clean, descriptive names in `match` statements:

```python
match get_user(1):
    case Ok(user):
        print(f"Found: {user.name}")
    case Err(error):
        print(f"Error: {error}")
```

---

## API & Philosophy

### Zero-Escape Safety
Crashing operations (panics) are strictly isolated in the `.unsafe` namespace. This ensures that "unwrapping" is always an intentional, visible choice in your codebase.

```python
res = Err("critical failure")

# This will raise a helpful AttributeError explaining the safe alternatives
res.unwrap() 

# Use the unsafe namespace if you specifically need to panic
res.unsafe.unwrap() # Raises UnwrapError
```

### Creating & Checking
```python
from result import Ok, Err, OkErr, is_ok, is_err

res1 = Ok(200)
res2 = Err(404)

isinstance(res1, OkErr) # True
is_ok(res1)             # True (TypeIs narrowing)
```

### Safe Conversion
Convert to optional values without risking exceptions:
```python
res = Ok(10)
val = res.ok()  # 10
err = res.err() # None

res_err = Err("fail")
val = res_err.ok()  # None
err = res_err.err() # "fail"
```

### Functional Chaining
```python
# Chaining success paths
Ok(10).map(lambda x: x * 2).tap(print).and_then(lambda x: Ok(x + 1))

# Recovery paths
Err("fail").or_else(lambda _: Ok(0))

# Side effects
res.tap(log_success).tap_err(log_failure)
```

---

## Modern Features

### `@safe` Decorator
Quickly lift existing exception-throwing code into `Result` containers. Supports both synchronous and asynchronous functions with perfect type inference.

```python
from result import safe

@safe(ValueError)
def parse(s: str) -> int:
    return int(s)

parse("10")           # Ok(10)
parse("not a number") # Err(ValueError(...))
```

### Do-Notation (Monadic Bind)
Eliminate nested "if is_ok" blocks with imperative-style generator syntax.

```python
from result import do_notation, Ok

@do_notation
def process_data(user_id: int):
    user = yield fetch_user(user_id)      # Auto-unwraps Ok or short-circuits Err
    profile = yield fetch_profile(user)   # Type of 'user' is the success value
    return profile.avatar_url             # Automatically wrapped in Ok
```

### Async Integration
First-class support for `async/await` throughout the API:
```python
await res.map_async(async_func)
await res.and_then_async(async_returning_result)

@do_notation_async
async def workflow():
    data = yield await async_step_1()
    yield Ok(data.summary)
```

---

## Testing Philosophy
The library uses a **Dual-Layered Suite** to ensure maximum reliability:
1.  **Public API Layer**: 100% type-safe with zero `type: ignore` comments. Ensures a perfect IDE experience for consumers.
2.  **Internal Layer**: Behavioral tests for educational safeguards and coverage mop-ups, pushing the type system to verify correct failure modes.
