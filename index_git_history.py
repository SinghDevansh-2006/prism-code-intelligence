"""Index real Python file history without checking out or executing repository code."""
import argparse
import json
import subprocess
from pathlib import Path
from src.versioned_retriever import VersionedSemanticIndex


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.PIPE)


def index_history(repo, paths, index, ref='HEAD', max_commits=20):
    if max_commits < 1:
        raise ValueError("max_commits must be positive")
    # Resolve ref first so a user-supplied value cannot become a git option.
    commit = git(repo, 'rev-parse', '--verify', '--end-of-options', ref + '^{commit}').decode().strip()
    for path in paths:
        if Path(path).is_absolute() or '..' in Path(path).parts or not path.endswith('.py'):
            raise ValueError('Use repository-relative Python paths without ..')
    commits = git(repo, 'rev-list', '--first-parent', '--reverse', f'--max-count={max_commits}', commit).decode().splitlines()
    initial_is_baseline = len(git(repo, "rev-list", "--parents", "-n", "1", commits[0]).split()) > 1 if commits else False
    snapshots_path = index.root / 'snapshots.json'
    snapshots = json.loads(snapshots_path.read_text()) if snapshots_path.exists() else {'commits': [], 'snapshots': {}, 'paths': sorted(paths)}
    if snapshots.get('paths') != sorted(paths):
        raise ValueError('Use the same selected paths or a new output index')
    def read_code(sha, path):
        entry = git(repo, 'ls-tree', '-z', sha, '--', path)
        if not entry:
            return None
        if not entry.startswith((b'100644 ', b'100755 ')):
            raise ValueError(f'Expected a regular Python file: {path}')
        blob = git(repo, 'show', f'{sha}:{path}')
        if len(blob) > 1_000_000:
            raise ValueError(f'File exceeds 1 MB safety limit: {path}')
        return blob.decode('utf-8')

    events = []
    previous = {}
    current = {}
    if snapshots['commits']:
        last = snapshots['commits'][-1]
        previous = {path: read_code(last, path) for path in paths}
        current = dict(snapshots['snapshots'][last])
    for sha in commits:
        if sha in snapshots['snapshots']:
            previous = {path: read_code(sha, path) for path in paths}
            current = dict(snapshots['snapshots'][sha])
            continue
        if snapshots['commits'] and sha not in snapshots['snapshots']:
            ancestry = subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', snapshots['commits'][-1], sha],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if ancestry.returncode != 0:
                raise ValueError('Cannot append older or divergent snapshots; use a new output directory')
        for path in paths:
            code = read_code(sha, path)
            old = previous.get(path)
            previous[path] = code
            if old == code:
                continue
            if code is None:
                current.pop(path, None)
            else:
                current[path] = sha
            if (path, sha) in index.existing_keys:
                continue
            lineage = index.history(path)
            if lineage:
                latest = lineage[-1].get('metadata', {}).get('commit')
                if not latest:
                    raise ValueError('Use a dedicated Git history output index')
                ancestry = subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', latest, sha],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if ancestry.returncode != 0:
                    raise ValueError('Cannot append older or divergent history; use a new output directory')
            operation = 'deleted'  if code is None else ('added' if old is None else 'modified')
            if sha == commits[0] and initial_is_baseline and code is not None and not lineage:
                operation = 'baseline'
            update = index.add_version(path, sha, code or '', title=path,
                metadata={'commit': sha, 'path': path, 'operation': operation,
                          'history_scope': 'bounded first-parent history; initial entry is a baseline'})
            events.append({'commit': sha, 'path': path, 'operation': operation, **update})
        snapshots['snapshots'][sha] = dict(current)
        if sha not in snapshots['commits']:
            snapshots['commits'].append(sha)
        temporary = snapshots_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(snapshots, indent=2)+'\n')
        temporary.replace(snapshots_path)
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--path', action='append', required=True)
    parser.add_argument('--ref', default='HEAD')
    parser.add_argument('--max-commits', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.max_commits < 1:
        parser.error('--max-commits must be positive')
    index = VersionedSemanticIndex(args.output, device='cpu')
    events = index_history(args.repo, args.path, index, args.ref, args.max_commits)
    print(json.dumps({'events': events, 'stats': index.stats()}, indent=2))


if __name__ == '__main__':
    main()
