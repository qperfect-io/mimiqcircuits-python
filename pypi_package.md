# Steps to Publish `mimiqcircuits` on PyPI

1. Set Project Description:
   description = "MimiqCircuits is a quantum computing library designed to handle advanced quantum circuit operations."

2. Install Poetry (if not already installed)

```bash
   curl -sSL https://install.python-poetry.org | python3 -
```

3. Test the Build Locally

```bash
   poetry build
```

This command will create a dist/ directory with .tar.gz and .whl files for your project.

4. Then create a PyPI Account:
Create an account on PyPI.

5. Publish Package on PyPI:

```bash

poetry config pypi-token.pypi `api-token`

```

Generate an API token by visiting your PyPI account settings.

After configuring the token, run

```bash
poetry publish --build
```

6. Install and Verify:

```python
   pip install mimiqcircuits
```

7. Add Badges (Optional):
After publishing, update your README.md to include a PyPI version badge and installation instructions to make it easier for users.

## Additional Notes

- Version Management:
   Make sure you increment the version in your `pyproject.toml` each time you make significant changes (e.g., version = "0.10.1").

- Documentation:
   Since you are using Sphinx under the `[tool.poetry.group.docs]` section, you may also want to publish your documentation online using something like ReadTheDocs.

## Example `.yml` for GitHub Actions

name: Publish Python Package

```yml
on:
  push:
    tags:
      - 'v*.*.*'  # This triggers the action only when a version tag is pushed (e.g., v0.10.0)

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3  # Check out the code

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'  # Set your Python version

      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          poetry config virtualenvs.create false
          poetry install --no-interaction --no-ansi

      - name: Build Package
        run: |
          poetry build

      - name: Publish to PyPI
        env:
          PYPI_TOKEN: ${{ secrets.PYPI_API_TOKEN }}  # Store your PyPI token as a secret in GitHub
        run: |
          poetry publish --build --username __token__ --password $PYPI_TOKEN
```

## Version Management

Update the version in your `pyproject.toml` file:

version = "0.10.1"

- **Patch version**: Increment for bug fixes (e.g., 0.10.1 → 0.10.2).
- **Minor version**: Increment for new features that are backward-compatible (e.g., 0.10.0 → 0.11.0).
- **Major version**: Increment for breaking changes (e.g., 0.10.0 → 1.0.0).

Rebuild and publish the updated package:

```bash
poetry build

poetry publish --build
```
Ensure users get the updated version:

```bash
pip install --upgrade mimiqcircuits
```