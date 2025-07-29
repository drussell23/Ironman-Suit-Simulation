# Adaptive AI Test Suite

This directory contains comprehensive tests for the Iron Man Suit's Adaptive AI System.

## Test Structure

### Unit Tests
- `test_reinforcement_learning.py` - Tests for RL agents (DQN, PPO, SAC)
- `test_tactical_decision.py` - Tests for tactical decision making and threat assessment
- `test_behavioral_adaptation.py` - Tests for pilot behavior modeling and adaptation
- `test_predictive_analytics.py` - Tests for predictive analytics and performance optimization
- `test_cognitive_load.py` - Tests for cognitive load management and automation control
- `test_ai_system.py` - Tests for the main AI system integration

### Integration Tests
- `test_integration.py` - Comprehensive integration tests for the complete system

## Running Tests

### Run all tests
```bash
cd backend/adaptive_ai
python -m pytest tests/
```

### Run specific test file
```bash
python -m pytest tests/test_reinforcement_learning.py -v
```

### Run with coverage
```bash
python -m pytest tests/ --cov=adaptive_ai --cov-report=html
```

### Run only unit tests
```bash
python -m pytest tests/ -m unit
```

### Run only integration tests
```bash
python -m pytest tests/ -m integration
```

## Test Categories

### Fast Tests
Most unit tests run quickly and test individual components in isolation.

### Slow Tests
Some tests, particularly integration tests and learning tests, may take longer:
- Long duration stability tests
- Multi-agent coordination tests
- Extended learning cycles

### Tests Requiring Dependencies
Some tests require specific libraries:
- Tests marked with `@pytest.mark.skipif(not torch)` require PyTorch
- Tests marked with `@pytest.mark.skipif(not SKLEARN_AVAILABLE)` require scikit-learn

## Writing New Tests

When adding new tests:
1. Follow the existing naming conventions
2. Use appropriate fixtures from `conftest.py`
3. Mark tests appropriately (unit, integration, slow, etc.)
4. Ensure tests are independent and can run in any order
5. Mock external dependencies appropriately
6. Include docstrings explaining what each test verifies

## Test Coverage Goals

We aim for:
- >90% coverage for core AI algorithms
- >85% coverage for decision-making logic
- >80% coverage for integration points
- 100% coverage for safety-critical functions

## Continuous Integration

These tests are designed to run in CI/CD pipelines with:
- Parallel test execution support
- Proper cleanup of resources
- Timeout handling for long-running tests
- Clear error reporting