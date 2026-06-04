# contributing to yabplot

thank you for your interest in contributing to `yabplot`! we welcome contributions from the neuroimaging and open-source communities.

## how to contribute

### reporting bugs
- use the [github issue tracker](https://github.com/teanijarv/yabplot/issues) to report bugs.
- include a minimal reproducible example and information about your environment (os, python version, package versions).

### suggesting enhancements
- open an issue describing the proposed feature and why it would be useful for the community.

### pull requests
1. fork the repository and create your branch from `main`.
2. if you've added code that should be tested, add tests in the `tests/` directory.
3. if you've changed API, update the documentation.
4. ensure the test suite passes: `uv run pytest tests/`
5. submit a pull request with a clear description of the changes.

## development setup

we use `uv` for dependency management. to set up your development environment:

```bash
git clone https://github.com/teanijarv/yabplot.git
cd yabplot
uv venv
uv pip install -e ".[docs]"
uv pip install pytest
```

## community
by contributing, you agree to abide by our [code of conduct](CODE_OF_CONDUCT.md).
