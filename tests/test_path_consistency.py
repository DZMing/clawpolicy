#!/usr/bin/env python3
"""Test default path consistency across all entry points."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lib.api import ConfirmationAPI
from lib.cli import ClawPolicyCLI
from lib.integration import IntentAlignmentEngine
from lib.paths import get_local_config_path, resolve_local_config_path


class TestDefaultPathConsistency:
    """Ensure all components use the same default state directory."""

    def test_default_local_config_path_is_cwd_clawpolicy(self) -> None:
        """Default config should be at .clawpolicy/config.json in current directory."""
        expected = Path.cwd() / ".clawpolicy" / "config.json"
        actual = get_local_config_path()
        assert actual == expected

    def test_resolve_local_config_without_args_returns_cwd_clawpolicy(self) -> None:
        """resolve_local_config_path() should return cwd/.clawpolicy/config.json."""
        expected = Path.cwd() / ".clawpolicy" / "config.json"
        actual = resolve_local_config_path()
        assert actual == expected

    def test_resolve_local_config_with_explicit_path(self, tmp_path: Path) -> None:
        """resolve_local_config_path() should respect explicit path."""
        custom_config = tmp_path / "custom" / "config.json"
        actual = resolve_local_config_path(custom_config)
        assert actual == custom_config.expanduser()

    def test_confirmation_api_default_memory_dir_is_cwd_clawpolicy(self, tmp_path: Path) -> None:
        """ConfirmationAPI should default to cwd/.clawpolicy when no path given."""
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            api = ConfirmationAPI()  # No memory_dir argument
            expected = tmp_path / ".clawpolicy"
            assert api.memory_dir == expected
        finally:
            os.chdir(original_cwd)

    def test_confirmation_api_explicit_memory_dir(self, tmp_path: Path) -> None:
        """ConfirmationAPI should respect explicit memory_dir."""
        custom_dir = tmp_path / "custom_memory"
        api = ConfirmationAPI(memory_dir=custom_dir)
        assert api.memory_dir == custom_dir.expanduser()

    def test_cli_get_memory_dir_default_to_cwd_clawpolicy(self) -> None:
        """CLI should default to cwd/.clawpolicy."""
        cli = ClawPolicyCLI()
        expected = Path.cwd() / ".clawpolicy"
        actual = cli.get_memory_dir()
        assert actual == expected

    def test_cli_get_memory_dir_with_cwd_arg(self, tmp_path: Path) -> None:
        """CLI should respect cwd argument."""
        cli = ClawPolicyCLI()
        expected = tmp_path / ".clawpolicy"
        actual = cli.get_memory_dir(cwd=tmp_path)
        assert actual == expected

    def test_intent_alignment_engine_default_config_is_local_clawpolicy(self, tmp_path: Path) -> None:
        """IntentAlignmentEngine should default to repo/.clawpolicy/config.json."""
        engine = IntentAlignmentEngine(repo_path=str(tmp_path))
        expected = tmp_path / ".clawpolicy" / "config.json"
        assert engine.config_path == expected

    def test_intent_alignment_engine_explicit_config(self, tmp_path: Path) -> None:
        """IntentAlignmentEngine should respect explicit config_path."""
        custom_config = tmp_path / "custom_config.json"
        engine = IntentAlignmentEngine(
            repo_path=str(tmp_path),
            config_path=custom_config,
        )
        assert engine.config_path == custom_config.expanduser()


class TestReadOnlyCommandsDoNotWriteDisk:
    """Ensure read-only status commands don't modify disk state."""

    def test_status_command_is_readonly(self, tmp_path: Path, capsys) -> None:
        """
        status command should NOT create any directory or file.

        This is the key fix: read-only operations should have zero side effects.
        """
        cli = ClawPolicyCLI()
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            # Don't call init(), just call status() directly
            cli.status()
        finally:
            os.chdir(original_cwd)

        # Verify NOTHING was created
        clawpolicy_dir = tmp_path / ".clawpolicy"
        assert not clawpolicy_dir.exists(), "status command should not create .clawpolicy directory"

        # Verify no files were created
        assert not (tmp_path / ".clawpolicy" / "config.json").exists()
        assert not (tmp_path / ".clawpolicy" / "policy").exists()

    def test_policy_status_command_is_readonly(self, tmp_path: Path) -> None:
        """policy status subcommand should not create any directory or file."""
        cli = ClawPolicyCLI()
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cli.policy_status()
        finally:
            os.chdir(original_cwd)

        # Verify NOTHING was created
        clawpolicy_dir = tmp_path / ".clawpolicy"
        assert not clawpolicy_dir.exists(), "policy status should not create .clawpolicy directory"

    def test_confirmation_api_readonly_queries_dont_create_policy_store(self, tmp_path: Path) -> None:
        """
        ConfirmationAPI read-only queries should not create policy store.

        This is the key fix: instantiating ConfirmationAPI should not create
        directories or files for read-only use cases.
        """
        memory_dir = tmp_path / ".clawpolicy"
        api = ConfirmationAPI(memory_dir=memory_dir)

        # Instantiating ConfirmationAPI should NOT create anything
        assert not memory_dir.exists(), "ConfirmationAPI init should not create memory directory"
        assert not (memory_dir / "policy").exists()

        # Getting confidence history should still not create anything
        history = api.get_confidence_history()
        assert history["rules"] == []
        assert not memory_dir.exists(), "get_confidence_history should not create memory directory"

    def test_get_recent_decisions_does_not_create_files(self, tmp_path: Path) -> None:
        """get_recent_decisions() should not create any files."""
        memory_dir = tmp_path / ".clawpolicy"
        api = ConfirmationAPI(memory_dir=memory_dir)

        decisions = api.get_recent_decisions(limit=10)

        # Should return empty list, not create any files
        assert decisions["decisions"] == []
        assert not memory_dir.exists(), "get_recent_decisions should not create memory directory"

    def test_init_command_does_create_files(self, tmp_path: Path) -> None:
        """init command SHOULD create directories and files (by design)."""
        cli = ClawPolicyCLI()
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = cli.init()
            assert result is True
        finally:
            os.chdir(original_cwd)

        # Verify init DID create files
        clawpolicy_dir = tmp_path / ".clawpolicy"
        assert clawpolicy_dir.exists(), "init should create .clawpolicy directory"

        policy_dir = clawpolicy_dir / "policy"
        assert policy_dir.exists(), "init should create policy directory"

        assert (clawpolicy_dir / "config.json").exists(), "init should create config.json"
        assert (policy_dir / "rules.json").exists(), "init should create rules.json"
        assert (policy_dir / "playbooks.json").exists(), "init should create playbooks.json"
        assert (policy_dir / "policy_events.jsonl").exists(), "init should create policy_events.jsonl"
