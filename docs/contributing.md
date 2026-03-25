# Contributing

## Development Setup

```bash
# Clone the repository
git clone https://github.com/UmeAiRT/ComfyUI-UmeAiRT-Toolkit.git
cd ComfyUI-UmeAiRT-Toolkit

# Install dependencies
pip install -r requirements.txt

# Run tests
python run_tests.py

# Run with coverage
coverage run --source=modules run_tests.py
coverage report -m --skip-covered
```

## Testing

Every new node should have at least a structural test. Copy `tests/_template_node_test.py` and fill in your node class:

```python
def test_input_types(self):
    inputs = YourNode.INPUT_TYPES()
    self.assertIn("required", inputs)

def test_function_exists(self):
    node = YourNode()
    self.assertTrue(callable(getattr(node, YourNode.FUNCTION)))
```

## Documentation

Documentation lives in `docs/` and uses [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

```bash
# Preview locally
pip install mkdocs-material
mkdocs serve

# Build
mkdocs build --strict
```

Each node should have a documentation page in `docs/nodes/` following the standard template with I/O tables.

## Code Style

- Linted with [Ruff](https://docs.astral.sh/ruff/) (`ruff check modules/`)
- Security scanned with [Bandit](https://bandit.readthedocs.io/)
- CI enforces both on every push
