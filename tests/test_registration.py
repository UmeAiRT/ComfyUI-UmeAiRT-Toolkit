"""
UmeAiRT Toolkit - Node Registration Tests
--------------------------------------------
Validates that NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
are consistent and all classes have required attributes.
"""

import sys
import os
import re
import unittest

# Force UTF-8
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class TestNodeRegistration(unittest.TestCase):
    """Validates __init__.py registration consistency (static analysis)."""

    def setUp(self):
        init_path = os.path.join(PROJECT_ROOT, '__init__.py')
        with open(init_path, 'r', encoding='utf-8') as f:
            self.content = f.read()

        # Extract class mapping keys
        class_pat = re.compile(r'"(UmeAiRT_\w+)":\s*UmeAiRT_')
        self.class_keys = class_pat.findall(self.content)

        # Extract display name mapping keys
        display_section = self.content.split('NODE_DISPLAY_NAME_MAPPINGS')[1] if 'NODE_DISPLAY_NAME_MAPPINGS' in self.content else ''
        display_pat = re.compile(r'"(UmeAiRT_\w+)":\s*"')
        self.display_keys = display_pat.findall(display_section)

    def test_all_class_mappings_have_display_names(self):
        """Every entry in NODE_CLASS_MAPPINGS must have a NODE_DISPLAY_NAME_MAPPINGS entry."""
        missing = set(self.class_keys) - set(self.display_keys)
        if missing:
            self.fail(f"Missing display names for: {', '.join(sorted(missing))}")

    def test_no_orphan_display_names(self):
        """Every display name entry must have a corresponding class mapping."""
        orphans = set(self.display_keys) - set(self.class_keys)
        if orphans:
            self.fail(f"Orphan display names (no class): {', '.join(sorted(orphans))}")

    def test_no_duplicate_class_mappings(self):
        """No class should be registered twice."""
        seen = set()
        dupes = []
        for key in self.class_keys:
            if key in seen:
                dupes.append(key)
            seen.add(key)
        if dupes:
            self.fail(f"Duplicate class mappings: {', '.join(dupes)}")

    def test_no_duplicate_display_names(self):
        """No two nodes should have the exact same display name (causes user confusion)."""
        display_pat = re.compile(r'"UmeAiRT_\w+":\s*"([^"]+)"')
        display_section = self.content.split('NODE_DISPLAY_NAME_MAPPINGS')[1] if 'NODE_DISPLAY_NAME_MAPPINGS' in self.content else ''
        names = display_pat.findall(display_section)

        seen = {}
        dupes = []
        for name in names:
            if name in seen:
                dupes.append(name)
            seen[name] = True

        if dupes:
            self.fail(f"Duplicate display names: {', '.join(dupes)}")

    def test_minimum_node_count(self):
        """Sanity check: we should have at least 25 registered nodes."""
        self.assertGreaterEqual(len(self.class_keys), 25,
                                f"Only {len(self.class_keys)} nodes registered, expected >= 25")


class TestDependencySync(unittest.TestCase):
    """Validates that requirements.txt and pyproject.toml are in sync."""

    def setUp(self):
        req_path = os.path.join(PROJECT_ROOT, 'requirements.txt')
        pyp_path = os.path.join(PROJECT_ROOT, 'pyproject.toml')

        with open(req_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Parse requirements: skip comments and empty lines, strip version specifiers
        self.req_deps = set()
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Strip version specifiers like >=2.3.0
                name = re.split(r'[><=!~]', line)[0].strip().lower().replace('-', '_')
                if name:
                    self.req_deps.add(name)

        with open(pyp_path, 'r', encoding='utf-8') as f:
            pyp_content = f.read()
        # Parse ALL dependencies (core + optional) from pyproject.toml
        dep_pat = re.compile(r'"([a-zA-Z0-9_-]+)')
        self.pyp_deps = set()
        # Core deps
        if 'dependencies' in pyp_content:
            for section_start in ['dependencies = [', 'dependencies=[']:
                if section_start in pyp_content:
                    section = pyp_content.split(section_start)[1].split(']')[0]
                    self.pyp_deps.update(d.lower().replace('-', '_') for d in dep_pat.findall(section))
        # Optional deps (all groups)
        if 'optional-dependencies' in pyp_content:
            opt_section = pyp_content.split('optional-dependencies]')[1] if 'optional-dependencies]' in pyp_content else ''
            # Stop at the next [section]
            next_section = re.search(r'\n\[(?!project\.optional)', opt_section)
            if next_section:
                opt_section = opt_section[:next_section.start()]
            self.pyp_deps.update(d.lower().replace('-', '_') for d in dep_pat.findall(opt_section)
                                 if d.lower() not in ('seedvr2', 'facedetailer', 'all'))

    def test_requirements_in_pyproject(self):
        """Every dep in requirements.txt must be in pyproject.toml."""
        missing = self.req_deps - self.pyp_deps
        if missing:
            self.fail(f"In requirements.txt but not pyproject.toml: {', '.join(sorted(missing))}")

    def test_pyproject_in_requirements(self):
        """Every dep in pyproject.toml should be in requirements.txt."""
        missing = self.pyp_deps - self.req_deps
        if missing:
            self.fail(f"In pyproject.toml but not requirements.txt: {', '.join(sorted(missing))}")


if __name__ == "__main__":
    unittest.main()
