"""
UmeAiRT Toolkit - Tooltip Completeness Test
----------------------------------------------
Ensures every INPUT_TYPES parameter across all node files has a tooltip.
This test parses Python source files directly to catch missing tooltips
without needing the ComfyUI runtime.
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

# Node module files to audit
NODE_FILES = [
    os.path.join(PROJECT_ROOT, 'modules', f) for f in [
        'block_inputs.py',
        'block_sampler.py',
        'block_loaders.py',
        'image_nodes.py',
        'logic_nodes.py',
        'model_nodes.py',
        'utils_nodes.py',
    ]
]

# Keys that are section headers, not actual inputs
SKIP_KEYS = {'required', 'optional', 'hidden', 'prompt', 'extra_pnginfo', 'unique_id'}

# Pattern: "key": ("TYPE", {options}) — captures key and checks for tooltip
INPUT_PAT = re.compile(r'^\s+"(\w+)"\s*:\s*\(')


class TestTooltipCompleteness(unittest.TestCase):
    """Ensures every input parameter in every node has a tooltip."""

    def test_all_inputs_have_tooltips(self):
        missing = []

        for fpath in NODE_FILES:
            basename = os.path.basename(fpath)
            if not os.path.exists(fpath):
                continue

            with open(fpath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                m = INPUT_PAT.match(line)
                if not m:
                    continue
                key = m.group(1)
                if key in SKIP_KEYS:
                    continue

                # Check this line and next for tooltip
                context = line
                if i + 1 < len(lines):
                    context += lines[i + 1]
                if i + 2 < len(lines):
                    context += lines[i + 2]

                if 'tooltip' not in context:
                    missing.append(f"{basename}:{i+1} [{key}]")

        if missing:
            msg = f"\n{len(missing)} input(s) missing tooltips:\n"
            msg += "\n".join(f"  - {m}" for m in missing[:20])
            if len(missing) > 20:
                msg += f"\n  ... and {len(missing) - 20} more"
            self.fail(msg)


if __name__ == "__main__":
    unittest.main()
