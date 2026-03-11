#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def repo_facts(repo_root: Path):
    pyproject = (repo_root / 'pyproject.toml').read_text()
    readme = (repo_root / 'README.md').read_text()
    license_truth = (repo_root / 'LICENSE_SOURCE_OF_TRUTH.md').read_text() if (repo_root / 'LICENSE_SOURCE_OF_TRUTH.md').exists() else ''
    version = None
    description = None
    for line in pyproject.splitlines():
        if line.startswith('version = '):
            version = line.split('=', 1)[1].strip().strip('"')
        if line.startswith('description = '):
            description = line.split('=', 1)[1].strip().strip('"')
    return {
        'version': version,
        'description': description,
        'readme_has_canonical_source': 'Canonical Source of Truth' in readme,
        'license_truth_file': bool(license_truth),
    }


def clawhub_facts(slug: str):
    proc = subprocess.run(['clawhub', 'inspect', slug, '--json'], capture_output=True, text=True)
    if proc.returncode != 0:
        return {'ok': False, 'error': proc.stderr.strip() or proc.stdout.strip()}
    raw = proc.stdout.strip()
    if raw.startswith('- Fetching skill'):
        raw = '\n'.join(raw.splitlines()[1:]).strip()
    data = json.loads(raw)
    skill = data.get('skill') or {}
    latest = data.get('latestVersion') or {}
    return {
        'ok': True,
        'slug': skill.get('slug'),
        'summary': skill.get('summary'),
        'latest_version': latest.get('version') or (skill.get('tags') or {}).get('latest'),
        'license': latest.get('license'),
        'changelog': latest.get('changelog'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', required=True)
    ap.add_argument('--slug', required=True)
    args = ap.parse_args()

    repo = repo_facts(Path(args.repo_root))
    hub = clawhub_facts(args.slug)
    report = {
        'repo': repo,
        'clawhub': hub,
        'checks': {}
    }
    if hub.get('ok'):
        summary = hub.get('summary') or ''
        description_prefix = (repo.get('description') or '')[:48]
        report['checks'] = {
            'version_matches': repo['version'] == hub.get('latest_version'),
            'summary_starts_with_repo_description': summary.startswith(description_prefix),
            'canonical_source_declared': repo['readme_has_canonical_source'] and repo['license_truth_file'],
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
