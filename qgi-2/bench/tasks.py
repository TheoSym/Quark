"""Verifiable coding tasks for the A/B comparison.

Each task is a small repo plus a command that decides, objectively, whether the
agent fixed it. No LLM judge: the test either passes or it does not.

The tasks are deliberately small and self-contained. The question this bench
answers is *"does the QGI-2 harness change how well the same model codes?"* —
so a task should be within reach of a 9B model on a good day and out of reach on
a bad one. Tasks nothing can do, or everything can do, measure nothing.

They also exercise the specific things QGI-2 does differently:

- `multi_file` needs a fact learned in one file applied in another, which is
  what the graph is supposed to carry.
- `find_and_fix` requires a search before an edit, so it needs at least two tool
  rounds — the path that did not exist before tool calls were wired.
- `stateful` asks a follow-up that depends on the previous turn, which is where
  dropping history and relying on the graph either works or does not.
"""

from dataclasses import dataclass, field


@dataclass
class Task:
    name: str
    # filename -> contents
    files: dict
    prompt: str
    # Shell command run inside the repo; exit 0 means solved.
    verify: str
    # Optional second turn, to test whether memory carried across turns.
    followup: str | None = None
    followup_verify: str | None = None
    tags: list = field(default_factory=list)


PYTEST = "python -m pytest -q"


TASKS = [
    Task(
        name="failing_test",
        tags=["single-file", "one-round"],
        files={
            "calc.py": (
                "def median(xs):\n"
                "    s = sorted(xs)\n"
                "    n = len(s)\n"
                "    # Bug: returns the wrong element for even-length input.\n"
                "    return s[n // 2]\n"
            ),
            "test_calc.py": (
                "from calc import median\n\n"
                "def test_odd():\n"
                "    assert median([3, 1, 2]) == 2\n\n"
                "def test_even():\n"
                "    assert median([4, 1, 3, 2]) == 2.5\n"
            ),
        },
        prompt="test_even in test_calc.py fails. Fix calc.py so both tests pass. Do not edit the tests.",
        verify=PYTEST,
    ),
    Task(
        name="find_and_fix",
        tags=["search", "multi-round"],
        files={
            "app/__init__.py": "",
            "app/util.py": (
                "def slugify(s):\n"
                "    return s.lower().replace(' ', '-')\n"
            ),
            "app/models.py": (
                "from app.util import slugify\n\n"
                "class Post:\n"
                "    def __init__(self, title):\n"
                "        self.title = title\n"
                "        self.slug = slugify(title)\n"
            ),
            "app/views.py": "from app.models import Post\n",
            "test_slug.py": (
                "from app.models import Post\n\n"
                "def test_strips_punctuation():\n"
                "    assert Post('Hello, World!').slug == 'hello-world'\n\n"
                "def test_collapses_spaces():\n"
                "    assert Post('a   b').slug == 'a-b'\n"
            ),
        },
        prompt=(
            "test_slug.py fails. Find where slugs are produced and fix it so both "
            "tests pass. Do not edit the tests."
        ),
        verify=PYTEST,
    ),
    Task(
        name="multi_file",
        tags=["cross-file", "multi-round"],
        files={
            "store.py": (
                "RATE_LIMIT = 100\n\n"
                "class Store:\n"
                "    def __init__(self):\n"
                "        self.calls = 0\n\n"
                "    def get(self, key):\n"
                "        self.calls += 1\n"
                "        return None\n"
            ),
            "client.py": (
                "from store import Store\n\n"
                "class Client:\n"
                "    def __init__(self):\n"
                "        self.store = Store()\n\n"
                "    def fetch(self, key):\n"
                "        return self.store.get(key)\n"
            ),
            "test_limit.py": (
                "import pytest\n"
                "from client import Client\n"
                "from store import RATE_LIMIT\n\n"
                "def test_raises_over_limit():\n"
                "    c = Client()\n"
                "    for _ in range(RATE_LIMIT):\n"
                "        c.fetch('k')\n"
                "    with pytest.raises(RuntimeError):\n"
                "        c.fetch('k')\n"
            ),
        },
        prompt=(
            "test_limit.py fails. store.py defines RATE_LIMIT. Make Client.fetch "
            "raise RuntimeError once the store has served RATE_LIMIT calls. "
            "Do not edit the tests."
        ),
        verify=PYTEST,
    ),
    Task(
        name="stateful",
        tags=["memory", "two-turn"],
        files={
            "geometry.py": (
                "import math\n\n"
                "def circle_area(r):\n"
                "    return math.pi * r * r\n"
            ),
            "test_geometry.py": (
                "from geometry import circle_area\n\n"
                "def test_area():\n"
                "    assert round(circle_area(2), 4) == 12.5664\n"
            ),
        },
        prompt=(
            "Read geometry.py and tell me what the module does. Do not change anything yet."
        ),
        verify="python -c \"import geometry\"",
        # The follow-up never names the file. An agent that kept the first turn
        # in context, or learned a fact about it, can act; one that dropped both
        # has to go looking again.
        followup=(
            "Now add a function to that same module for the perimeter, following "
            "the style you just described, and a test for it."
        ),
        followup_verify=PYTEST,
    ),
]


def by_name(names):
    if not names:
        return TASKS
    wanted = set(names)
    missing = wanted - {t.name for t in TASKS}
    if missing:
        raise SystemExit(f"unknown task(s): {', '.join(sorted(missing))}")
    return [t for t in TASKS if t.name in wanted]
