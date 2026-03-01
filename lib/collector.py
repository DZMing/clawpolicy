#!/usr/bin/env python3
"""
data collection module - fromGitCollection of preference data from history and user actions
"""

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class GitPreferenceCollector:
    """fromGitHistory Collection Technology Preferences"""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.preferences: Dict[str, Any] = {
            "tech_stack": {},
            "file_types": {},
            "commit_patterns": {},
            "workflow": {}
        }

    def collect(self, max_commits: int = 100) -> Dict[str, Any]:
        """collectGitHistorical preference data"""
        print(f"📊 AnalyzingGithistory（recent{max_commits}commits）...")

        # Get commit history
        commits = self._get_commits(max_commits)
        print(f"✅ analyzed {len(commits)} commits")

        # Analysis Technology Stack
        self.preferences["tech_stack"] = self._analyze_tech_stack(commits)

        # Analyze file types
        self.preferences["file_types"] = self._analyze_file_types(commits)

        # Analytical workflow patterns
        self.preferences["workflow"] = self._analyze_workflow(commits)

        # Add metadata
        self.preferences["metadata"] = {
            "collected_at": datetime.now().isoformat(),
            "repo_path": str(self.repo_path),
            "commits_analyzed": len(commits),
            "confidence": self._calculate_confidence()
        }

        return self.preferences

    def _get_commits(self, max_count: int) -> List[Dict[str, Any]]:
        """GetGitCommit history"""
        try:
            result = subprocess.run(
                ["git", "log", f"-{max_count}", "--pretty=format:%H|%s|%an", "--name-only"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                print("⚠️  Unable to obtainGithistory")
                return []

            commits: List[Dict[str, Any]] = []
            lines = result.stdout.strip().split('\n')

            current_commit: Dict[str, Any] | None = None
            for line in lines:
                if '|' in line:  # Submit information line
                    parts = line.split('|')
                    current_commit = {
                        "hash": parts[0],
                        "subject": parts[1],
                        "author": parts[2],
                        "files": []
                    }
                    commits.append(current_commit)
                elif current_commit and line:  # file line
                    current_commit["files"].append(line)

            return commits

        except subprocess.TimeoutExpired:
            print("⚠️  GitCommand timeout")
            return []
        except Exception as e:
            print(f"⚠️  GetGithistorical failure: {e}")
            return []

    def _analyze_tech_stack(self, commits: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze technology stack preferences"""
        tech_stack = {
            "python": 0,
            "javascript": 0,
            "typescript": 0,
            "react": 0,
            "vue": 0,
            "fastapi": 0,
            "node": 0
        }

        for commit in commits:
            for file_path in commit.get("files", []):
                file_lower = file_path.lower()

                # DetectionPython
                if file_path.endswith('.py'):
                    tech_stack["python"] += 1

                # DetectionJavaScript/TypeScript
                if file_path.endswith('.js'):
                    tech_stack["javascript"] += 1
                if file_path.endswith('.ts') or file_path.endswith('.tsx'):
                    tech_stack["typescript"] += 1

                # DetectionReact
                if "react" in file_lower or file_path.endswith('.jsx'):
                    tech_stack["react"] += 1

                # DetectionVue
                if "vue" in file_lower:
                    tech_stack["vue"] += 1

                # DetectionFastAPI
                if "fastapi" in file_lower:
                    tech_stack["fastapi"] += 1

                # DetectionNode
                if "package.json" in file_path:
                    tech_stack["node"] += 1

        return tech_stack

    def _analyze_file_types(self, commits: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze file type preferences"""
        file_types: Dict[str, int] = {}

        for commit in commits:
            for file_path in commit.get("files", []):
                ext = Path(file_path).suffix or "(No suffix)"
                file_types[ext] = file_types.get(ext, 0) + 1

        return file_types

    def _analyze_workflow(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analytical workflow patterns"""
        workflow = {
            "test_first": False,
            "commit_frequency": {},
            "pair_programming": False
        }

        # Check if there is a test-driven development model
        test_commits = [c for c in commits if any(
            "test" in f.lower() for f in c.get("files", [])
        )]

        if len(test_commits) > len(commits) * 0.3:
            workflow["test_first"] = True
            workflow["test_ratio"] = len(test_commits) / len(commits)

        return workflow

    def _calculate_confidence(self) -> float:
        """Calculate confidence"""
        total_commits = sum(self.preferences["tech_stack"].values())
        if total_commits < 10:
            return 0.3  # Not enough data，low confidence
        elif total_commits < 50:
            return 0.7  # medium confidence
        else:
            return 0.95  # high confidence


def main():
    """Test data collection"""
    collector = GitPreferenceCollector()
    preferences = collector.collect()

    print("\n📊 learning results：")
    print("Technology stack preferences：")
    for tech, count in sorted(preferences["tech_stack"].items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  - {tech}: {count}Second-rate")

    print("\nFile type preference：")
    for ext, count in sorted(preferences["file_types"].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {ext}: {count}Second-rate")

    print(f"\nConfidence: {preferences['metadata']['confidence']*100}%")


if __name__ == "__main__":
    main()
