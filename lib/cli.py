#!/usr/bin/env python3
"""
OpenClaw Alignment command-line interface.

Provides one-command initialization and local memory configuration management.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional


class OpenClawAlignmentCLI:
    """Main OpenClaw Alignment CLI class."""

    def __init__(self):
        self.memory_dir_name = ".openclaw_memory"
        self.config_file_name = "config.json"
        self.templates = {
            "USER.md": "USER_template.md",
            "SOUL.md": "SOUL_template.md",
            "AGENTS.md": "AGENTS_template.md",
        }

    def get_template_dir(self) -> Path:
        """Return the packaged template directory."""
        # Resolve templates from the installed package
        package_dir = Path(__file__).parent
        template_dir = package_dir.parent / "templates"
        return template_dir

    def get_memory_dir(self, cwd: Optional[Path] = None) -> Path:
        """Return the memory directory path."""
        if cwd is None:
            cwd = Path.cwd()
        return cwd / self.memory_dir_name

    def init(self, target_dir: Optional[str] = None, force: bool = False) -> bool:
        """
        Initialize OpenClaw Alignment memory files.

        Args:
            target_dir: Target directory (defaults to current working directory)
            force: Overwrite existing files when True

        Returns:
            True if initialization succeeds
        """
        if target_dir:
            cwd = Path(target_dir).resolve()
        else:
            cwd = Path.cwd()

        memory_dir = self.get_memory_dir(cwd)
        template_dir = self.get_template_dir()

        # Validate template directory
        if not template_dir.exists():
            print(f"❌ Error: template directory not found: {template_dir}")
            print("   Make sure openclaw-alignment is installed correctly")
            return False

        # Check whether the memory directory already exists
        if memory_dir.exists():
            if not force:
                print(f"⚠️  Memory directory already exists: {memory_dir}")
                print("   Use --force to re-initialize")
                return False
            print("🔄 Forcing re-initialization...")
        else:
            print("🚀 Initializing OpenClaw Alignment memory...")

        # Create memory directory
        memory_dir.mkdir(parents=True, exist_ok=True)

        # Copy template files
        success_count = 0
        for target_name, template_name in self.templates.items():
            template_file = template_dir / template_name
            target_file = memory_dir / target_name

            if not template_file.exists():
                print(f"⚠️  Template file missing: {template_name}")
                continue

            if target_file.exists() and not force:
                print(f"⏭️  Skipped existing file: {target_name}")
                continue

            shutil.copy2(template_file, target_file)
            success_count += 1
            print(f"✅ Created: {target_file}")

        # Create config file
        config_file = memory_dir / self.config_file_name
        if not config_file.exists() or force:
            config = {
                "version": "1.0.0",
                "initialized_at": str(Path.cwd()),
                "memory_path": str(memory_dir),
                "features": {
                    "rl_enabled": True,
                    "auto_learning": True,
                    "safety_checks": True,
                },
            }
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"✅ Created: {config_file}")

        # Create .gitignore
        gitignore_file = memory_dir / ".gitignore"
        if not gitignore_file.exists() or force:
            with open(gitignore_file, "w", encoding="utf-8") as f:
                f.write("# OpenClaw Alignment local files\n")
                f.write("# Do not commit these files\n")
                f.write("config.json\n")
                f.write("*.backup\n")
                f.write("*.cache\n")
            print(f"✅ Created: {gitignore_file}")

        # Print success summary
        print("")
        print("=" * 60)
        print("✨ Initialization complete!")
        print("=" * 60)
        print(f"📂 Memory directory: {memory_dir}")
        print("📄 Created files:")
        for target_name in self.templates.keys():
            print(f"   - {target_name}")
        print(f"   - {self.config_file_name}")
        print("   - .gitignore")
        print("")
        print("📝 Next steps:")
        print("   1. Edit USER.md to define your personal preferences")
        print("   2. Review SOUL.md to confirm system principles")
        print("   3. Check AGENTS.md to see available tools")
        print("   4. Run: openclaw-align analyze")
        print("")

        return True

    def status(self) -> None:
        """Show current local status."""
        cwd = Path.cwd()
        memory_dir = self.get_memory_dir(cwd)
        config_file = memory_dir / self.config_file_name

        print("📊 OpenClaw Alignment status")
        print("")
        print(f"📂 Memory directory: {memory_dir}")
        print(f"   Status: {'✅ Exists' if memory_dir.exists() else '❌ Missing'}")
        print("")

        if memory_dir.exists():
            print("📄 Configuration files:")

            # Check config file
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                print(f"   ✅ {self.config_file_name}")
                print(f"      Version: {config.get('version', 'unknown')}")
                print(f"      RL enabled: {config.get('features', {}).get('rl_enabled', False)}")
            else:
                print(f"   ❌ {self.config_file_name} (missing)")

            # Check template files
            for target_name in self.templates.keys():
                target_file = memory_dir / target_name
                status = "✅" if target_file.exists() else "❌"
                print(f"   {status} {target_name}")
        else:
            print("💡 Tip: run 'openclaw-align init' to initialize memory files")
        print("")

    def version(self) -> None:
        """Show version info."""
        from . import __version__
        print(f"OpenClaw Alignment CLI v{__version__}")
        print("")
        print(f"Python: {__import__('sys').version}")
        print(f"Install path: {Path(__file__).parent.parent}")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="OpenClaw Alignment - reinforcement-learning driven alignment system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  openclaw-align init              Initialize memory files
  openclaw-align init --force      Force re-initialization
  openclaw-align init ~/projects   Initialize under a target directory
  openclaw-align status            Show current status
  openclaw-align version           Show version
        """
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize memory files"
    )
    init_parser.add_argument(
        "target_dir",
        nargs="?",
        help="Target directory (defaults to current directory)"
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing files"
    )

    # status command
    subparsers.add_parser(
        "status",
        help="Show current status"
    )

    # analyze command (keeps original behavior)
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze Git history and learn preferences"
    )
    analyze_parser.add_argument(
        "--repo",
        default=".",
        help="Git repository path"
    )
    analyze_parser.add_argument(
        "--commits",
        type=int,
        default=100,
        help="Number of commits to analyze"
    )

    args = parser.parse_args()
    cli = OpenClawAlignmentCLI()

    # Show version
    if args.version:
        cli.version()
        return

    # Execute command
    if args.command == "init":
        success = cli.init(
            target_dir=args.target_dir,
            force=args.force
        )
        exit(0 if success else 1)

    elif args.command == "status":
        cli.status()

    elif args.command == "analyze":
        # Call the existing analysis flow
        from .integration import IntentAlignmentEngine
        engine = IntentAlignmentEngine(args.repo)
        engine.run_analysis(args.commits)

    else:
        # Default behavior: print help
        parser.print_help()


if __name__ == "__main__":
    main()
