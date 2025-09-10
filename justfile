set dotenv-filename := x"${JUST_ENV:-.env}"
set dotenv-load
set dotenv-override

# sync packages with pyproject.toml
setup:
  uv sync --all-packages

# run tests
test:
  uv run pytest
  @echo "Tests passed"
