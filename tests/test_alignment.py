#!/usr/bin/env python3
"""
Test intent alignment functionality
"""

import sys
from pathlib import Path

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.integration import IntentAlignmentEngine


def test_collector():
    """Test data collection"""
    print("="*50)
    print("test1: data collection")
    print("="*50)

    engine = IntentAlignmentEngine(".")
    git_data = engine.collector.collect(50)

    assert git_data is not None, "Data collection failed"
    assert "tech_stack" in git_data, "Missing technology stack data"
    assert "metadata" in git_data, "Missing metadata"

    print("✅ Data collection test passed")
    print()


def test_learner():
    """Test learning algorithms"""
    print("="*50)
    print("test2: learning algorithm")
    print("="*50)

    # createengine
    from lib.learner import PreferenceLearner
    learner = PreferenceLearner()

    # simulated data
    mock_data = {
        "tech_stack": {
            "python": 10,
            "javascript": 5,
            "react": 8
        },
        "workflow": {
            "test_first": True,
            "test_ratio": 0.4
        },
        "metadata": {
            "confidence": 0.8
        }
    }

    preferences = learner.learn_from_git_history(mock_data)

    assert preferences is not None, "learning failure"
    assert "tech_stack" in preferences, "Missing technology stack preference"
    assert preferences["tech_stack"]["primary"] == "python", "Major technical identification errors"

    print("✅ Learning algorithm test passed")
    print()


def test_integration():
    """Test full integration"""
    print("="*50)
    print("test3: Full integration")
    print("="*50)

    engine = IntentAlignmentEngine(".")

    # Run analysis
    result = engine.run_analysis(10)

    assert result is not None, "Analysis failed"
    assert "tech_stack" in result, "Missing technology stack results"

    print("✅ Integration test passed")
    print()


def main():
    """Run all tests"""
    print("\n🧪 Start testing the intent alignment feature...\n")

    try:
        test_collector()
        test_learner()
        # test_integration()  # needGitstorehouse，Skip for now

        print("="*50)
        print("✅ All tests passed！")
        print("="*50)

    except AssertionError as e:
        print(f"❌ test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ mistake: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
