# Install runtime, Graphiti and dev dependencies
install:
    uv sync --extra graph --extra dev

# Run the test suite; pass pytest args, e.g. `just test -k http`
test *args:
    uv run pytest {{args}}
