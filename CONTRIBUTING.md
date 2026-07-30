# Contributing to Smart Portfolio GNN

Thank you for your interest in contributing to Smart Portfolio GNN! We welcome contributions from researchers, quantitative developers, and the open-source community.

## Development Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/USERNAME/smart-portfolio-gnn.git
   cd smart-portfolio-gnn
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   make install
   ```
   *This installs the package in editable mode along with testing tools.*

## Running Tests

Before submitting a pull request, ensure all tests pass:

```bash
make test
```

Our CI pipeline enforces a minimum test coverage of 70%.

## Code Style

We follow the standard Python formatting guidelines:
- **Black** for code formatting.
- **isort** for import sorting.
- **Flake8** for linting.

You can run the linters manually using:
```bash
black src tests dashboard scripts
isort src tests dashboard scripts
flake8 src tests dashboard scripts
```

## Submitting Pull Requests

1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes with clear, descriptive messages.
4. Push your branch and open a Pull Request against the `main` branch.
5. Ensure the GitHub Actions CI pipeline passes.
