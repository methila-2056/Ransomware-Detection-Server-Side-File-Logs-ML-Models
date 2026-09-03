# CONTRIBUTING

Thank you for your interest in contributing to the Ransomware Detection System!

## Getting Started

1. Fork the repository
2. Clone your fork
3. Install dependencies: `pip install -r requirements.txt`
4. Create a feature branch

## Development Workflow

1. Make your changes
2. Test locally: `python app.py`
3. Verify model training: `python ml_engine.py`
4. Validate database operations
5. Commit with clear messages
6. Push and create a pull request

## Code Style

- Follow PEP 8 for Python code
- Use descriptive variable names
- Add docstrings to all public functions/classes
- Keep functions focused and modular
- Use type hints where practical

## Commit Conventions

Use conventional commit messages:
```
feat: add new feature
fix: correct a bug
docs: update documentation
style: formatting changes
refactor: code restructure without behavior change
test: add/modify tests
build: build system changes
```

## Testing

Before submitting:
- Run the application and verify the dashboard loads
- Test both simulation and real monitoring modes
- Verify DB operations work correctly
- Ensure models load without errors

## Model Development

When changing ML models:
- Document hyperparameter changes
- Report before/after metrics
- Save updated models to `models/`
- Update `models/README.md`

## Version Control

- Never commit sensitive data
- Keep commits focused on a single concern
- Reference issues in commit messages when applicable
- Update `CHANGELOG.md` for significant changes

## Pull Request Guidelines

- Provide clear description of changes
- Reference related issues
- Include screenshots for UI changes
- Describe how to test the changes
