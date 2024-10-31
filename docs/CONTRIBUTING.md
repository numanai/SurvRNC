# Contributing to SurvRNC

Thank you for your interest in contributing to SurvRNC! We welcome contributions from the community and are grateful for your support.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Requests](#pull-requests)
- [Setting Up the Development Environment](#setting-up-the-development-environment)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming and inclusive environment for everyone.

## How to Contribute

1. **Fork the repository**: Click the "Fork" button at the top right corner of the repository page to create a copy of the repository in your GitHub account.

2. **Clone the repository**: Clone the forked repository to your local machine using the following command:

    ```bash
    git clone https://github.com/your-username/SurvRNC.git
    cd SurvRNC
    ```

3. **Create a new branch**: Create a new branch for your contribution using the following command:

    ```bash
    git checkout -b your-branch-name
    ```

4. **Make your changes**: Make the necessary changes to the codebase.

5. **Commit your changes**: Commit your changes with a descriptive commit message using the following command:

    ```bash
    git commit -m "Description of your changes"
    ```

6. **Push your changes**: Push your changes to your forked repository using the following command:

    ```bash
    git push origin your-branch-name
    ```

7. **Create a pull request**: Go to the original repository on GitHub and create a pull request from your forked repository.

## Code Style

Please follow the PEP 8 style guide for Python code. You can use tools like `flake8` and `black` to check and format your code.

## Testing

Ensure that your changes do not break existing tests and add new tests for any new functionality. You can run the tests using the following command:

```bash
pytest
```

## Pull Requests

When creating a pull request, please provide a clear and concise description of your changes. Include any relevant issue numbers and explain the purpose of the changes.

## Setting Up the Development Environment

1. **Create a virtual environment**: Create a virtual environment to isolate your development environment using the following command:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. **Install dependencies**: Install the required dependencies using the following command:

    ```bash
    pip install -r requirements.txt
    ```

3. **Install pre-commit hooks**: Install pre-commit hooks to ensure code quality using the following command:

    ```bash
    pre-commit install
    ```

4. **Run the tests**: Run the tests to ensure that everything is set up correctly using the following command:

    ```bash
    pytest
    ```

Thank you for contributing to SurvRNC!
