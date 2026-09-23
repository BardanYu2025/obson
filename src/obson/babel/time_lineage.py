"""Later-time provenance, including the coverage audit's named manifest.

Separate from the historical holdout scanner so its pinned source stays intact.
The graph builder also supports offline replay of exported manifest bytes.
"""
from pathlib import Path

from .dual_state import sha256
from .holdout_audit import PATH_KEYS, manifest_links, read_json, records

MANIFEST_NAMES = ('manifest.json', 'audit_manifest.json')


def manifest_paths(registry, exclude):
    return sorted({p.resolve() for name in MANIFEST_NAMES for p in registry.rglob(name)
                   if exclude not in p.resolve().parents})


def trace_lineage(manifests, hashes, registry, start, issues=()):
    """Resolve every declared manifest hash; named audits receive no exemption."""
    issues = list(issues); by_hash = {}
    for path, digest in hashes.items():
        by_hash.setdefault(digest, []).append(path)
    for paths in by_hash.values():
        paths.sort()
    queue = [str(start/'manifest.json')]; visited = set(); roots = []; graph = {}
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        visited.add(path)
        if path not in manifests:
            issues.append(dict(path=path, reason='ancestor_not_in_registry'))
            continue
        obj = manifests[path]; edges = []
        # A coverage audit stores paths in a dedicated mapping. Raw root/output
        # paths are not ancestors; its long/cross hashes must still resolve.
        mappings = [obj]
        if Path(path).name == 'audit_manifest.json' and isinstance(obj.get('paths'), dict):
            mappings.append(obj['paths'])
        for mapping in mappings:
            for key, value in mapping.items():
                if key not in PATH_KEYS or not isinstance(value, str):
                    continue
                target = Path(value)
                if not target.is_absolute():
                    issues.append(dict(path=path, reason='ambiguous_relative_ancestor', value=value))
                    continue
                if target.name in MANIFEST_NAMES:
                    edge = str(target.resolve())
                else:
                    options = [str((target/name).resolve()) for name in MANIFEST_NAMES]
                    edge = next((p for p in options if p in manifests), options[0])
                edges.append(edge)
        for digest in sorted(set(manifest_links(obj))):
            if digest in by_hash:
                edges.append(by_hash[digest][0])
            else:
                issues.append(dict(path=path, reason='unresolved_manifest_hash', sha256=digest))
        if records(obj):
            roots.append(path)
        if not edges and not records(obj):
            issues.append(dict(path=path, reason='opaque_ancestry_without_data_sources'))
        graph[path] = sorted(set(edges)); queue.extend(graph[path])

    def reaches_source(path, seen):
        if path in roots:
            return True
        if path in seen:
            return False
        return any(reaches_source(edge, seen | {path}) for edge in graph.get(path, []))

    for path in sorted(visited):
        if not reaches_source(path, set()):
            issues.append(dict(path=path, reason='ancestor_has_no_resolved_dataset_path'))
    source_records = [r for obj in manifests.values() for r in records(obj)]
    if not roots or not source_records:
        issues.append(dict(reason='no_dataset_source_manifest'))
    return dict(registry=str(registry), manifests=hashes, ancestors=sorted(visited),
                dataset_roots=sorted(roots), edges=graph, issues=issues,
                source_records=source_records, verified=not issues,
                manifest_names=list(MANIFEST_NAMES),
                scope='Available registered manifests and named coverage audits only; deleted/external or unrecorded human research cannot be certified.')


def scan_lineage(registry, start, exclude):
    registry, start, exclude = (Path(p).resolve() for p in (registry,start,exclude))
    manifests = {}; hashes = {}; issues = []
    for path in manifest_paths(registry,exclude):
        try:
            obj = read_json(path)
            if not isinstance(obj, dict):
                raise ValueError('Manifest must be a JSON object')
            manifests[str(path)] = obj; hashes[str(path)] = sha256(path)
        except (OSError, ValueError, TypeError) as exc:
            issues.append(dict(path=str(path),reason='unreadable_registered_manifest',detail=str(exc)))
    return trace_lineage(manifests,hashes,registry,start,issues)
