# aliases

PYSOURCE := "."

# main check (Enforced before commit)

format:
    ruff format --preview {{ PYSOURCE }}

ruff-check:
    ruff check --fix --unsafe-fixes {{ PYSOURCE }}

basedpyright-check:
    basedpyright {{ PYSOURCE }}

mypy-check:
    mypy {{ PYSOURCE }}

ty-check:
    ty check {{ PYSOURCE }}

check: format ruff-check ty-check mypy-check basedpyright-check

test:
    pytest -v tests/

coverage:
    pytest -v --cov --cov-report html

# Documentation

docs-build:
    uv run mkdocs build

docs-serve:
    uv run mkdocs serve

docs-deploy:
    uv run mkdocs gh-deploy --force

# Additional analysis checks and Tasks (not Enforced)

clean:
    ruff clean
    rm -rf .pytest_cache/
    rm -rf .coverage
    rm -rf htmlcov/
    rm -rf __pycache__/
    rm -rf .mypy_cache/
    rm -rf .pytest_cache/
    rm -rf .ropeproject/
    rm -rf .hypothesis/
    rm -rf site/

clean-deps:
    rm -rf .venv/

clean-all: clean clean-deps

radon:
    radon cc -a -nc -s {{ PYSOURCE }}

radon-mi:
    radon mi -s {{ PYSOURCE }}

vulture:
    vulture {{ PYSOURCE }} --min-confidence 60 --sort-by-size --exclude .venv/

check-uv-lock:
    [ -f ./uv.lock ] && uv lock --check || echo "No uv.lock file found, skipping lock check"

compile-user-dep:
    uv pip compile pyproject.toml -o requirements.txt

compile-dev-dep:
    uv pip compile pyproject.toml --group dev -o requirements-dev.txt

compile-dep: compile-user-dep compile-dev-dep check-uv-lock
