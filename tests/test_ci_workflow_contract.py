"""Contract gates for ``.github/workflows/tests.yml`` (m4787).

Why this file exists
--------------------
``main@26a32789`` reported ``TensorDG tests / test`` as red with the conclusion
``cancelled``, **zero steps executed** and ``runner_id = 0``.  The job asked for
``runs-on: ci-loong64``; the two repo-scoped listeners ``node099-ci-dg-1/2``
stayed in "Listening for Jobs" for seven hours and never received a job, i.e.
the *only* thing that went wrong was the label set the job requested.  Nothing
in the test suite was involved.

These gates are executable, not greps-in-spirit:

* the ``concurrency`` expression of the workflow is *rendered* with a small
  evaluator for the subset GitHub uses here (``==``, ``&&``, ``||``,
  ``format()``) and the resulting group names are asserted;
* the ``runs-on`` request is matched against the labels a runner registration
  always publishes (``self-hosted`` + OS) by a matcher, and the previous value
  is used as a **negative control** so the assertion cannot pass vacuously.

No third-party YAML/pytest plugin is required: the file is read as text, which
is all these gates need (and keeps the gate runnable in a bare CI image).
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "tests.yml"

# Labels every GitHub Actions self-hosted runner publishes unless registration
# explicitly disables them.  The DG listeners were registered in the same batch
# as the other repositories' runners, so this is the label set they carry.
DEFAULT_SELF_HOSTED_LABELS = {"self-hosted", "linux"}


def _workflow_text() -> str:
    assert WORKFLOW.is_file(), f"missing workflow: {WORKFLOW}"
    return WORKFLOW.read_text(encoding="utf-8")


def _expression_value(line: str, key: str) -> str:
    """Return the value that follows ``key:`` (a template with ``${{ ... }}``)."""
    match = re.search(rf"^\s*{key}:\s*(.+?)\s*$", line, re.MULTILINE)
    assert match, f"no expression found for {key!r}"
    return match.group(1)


def _render_template(value: str, context: dict[str, str]) -> str:
    """Render every ``${{ ... }}`` placeholder inside a plain YAML value."""
    return re.sub(
        r"\$\{\{.*?\}\}", lambda m: _render(m.group(0), context), value
    )


def _render(expression: str, context: dict[str, str]) -> str:
    """Render the GitHub-expression subset used by this workflow."""
    body = expression.strip()
    assert body.startswith("${{") and body.endswith("}}"), expression
    body = body[3:-2].strip()
    python = re.sub(r"github\.(\w+)", lambda m: f"ctx[{m.group(1)!r}]", body)
    python = re.sub(r"format\((\'[^\']*\'),\s*([^)]+)\)", r"(\1.format(\2))", python)
    python = python.replace("&&", " and ").replace("||", " or ")
    ctx = {k: v for k, v in context.items()}
    return str(eval(python, {"__builtins__": {}}, {"ctx": ctx}))  # noqa: S307 - fixed subset


def _runs_on_labels(text: str) -> list[str]:
    """Labels requested by the single job of this workflow."""
    match = re.search(r"^\s*runs-on:\s*(.+)$", text, re.MULTILINE)
    assert match, "workflow has no runs-on"
    raw = match.group(1).strip()
    if raw.startswith("["):
        return [item.strip().strip("'\"") for item in raw.strip("[]").split(",") if item.strip()]
    return [raw.strip("'\"")]


def _pool_matches(requested: list[str], runner_labels: set[str]) -> bool:
    """GitHub semantics: every requested label must be present on the runner."""
    return all(label.lower() in runner_labels for label in requested)


# ---------------------------------------------------------------- concurrency
def test_main_commits_get_distinct_concurrency_groups() -> None:
    """A newer merge must never cancel an older commit's pending main run."""
    expression = _expression_value(_workflow_text(), "group")
    first = _render_template(expression, {"workflow": "TensorDG tests", "ref": "refs/heads/main", "sha": "a" * 40})
    second = _render_template(expression, {"workflow": "TensorDG tests", "ref": "refs/heads/main", "sha": "b" * 40})
    assert first != second, "main is ref-scoped: two commits share one group and cancel each other"


def test_ref_wide_main_group_is_the_regression_this_guards() -> None:
    """Negative control: the pre-fix expression puts both commits in one group."""
    legacy = "${{ github.workflow }}-${{ github.ref }}"
    first = _render_template(legacy, {"workflow": "TensorDG tests", "ref": "refs/heads/main", "sha": "a" * 40})
    second = _render_template(legacy, {"workflow": "TensorDG tests", "ref": "refs/heads/main", "sha": "b" * 40})
    assert first == second, "the negative control must reproduce the regression"


def test_branch_and_pr_runs_stay_superseded() -> None:
    """Branches/PRs keep the ref-wide group, so a new push kills the stale run."""
    expression = _expression_value(_workflow_text(), "group")
    for ref in ("refs/heads/agent/m4787", "refs/pull/194/merge"):
        first = _render_template(expression, {"workflow": "TensorDG tests", "ref": ref, "sha": "a" * 40})
        second = _render_template(expression, {"workflow": "TensorDG tests", "ref": ref, "sha": "b" * 40})
        assert first == second, f"{ref} must stay ref-scoped"


def test_cancel_in_progress_never_kills_main() -> None:
    expression = _expression_value(_workflow_text(), "cancel-in-progress")
    main = _render_template(expression, {"ref": "refs/heads/main"})
    branch = _render_template(expression, {"ref": "refs/heads/agent/m4787"})
    assert main == "False", f"main must not be cancelled in progress, got {main}"
    assert branch == "True", f"branch pushes must supersede each other, got {branch}"


# --------------------------------------------------------------------- runner
def test_job_requests_labels_the_repo_runners_actually_publish() -> None:
    """The DG listeners carry the default labels; ask for those, not a custom one."""
    requested = _runs_on_labels(_workflow_text())
    assert _pool_matches(requested, DEFAULT_SELF_HOSTED_LABELS), (
        f"runs-on {requested} cannot be satisfied by a runner labelled "
        f"{sorted(DEFAULT_SELF_HOSTED_LABELS)} (this is how main@26a32789 got "
        "cancelled with runner_id=0)"
    )


def test_custom_only_label_is_rejected_by_the_same_matcher() -> None:
    """Negative control: the pre-fix request fails the matcher above."""
    assert not _pool_matches(["ci-loong64"], DEFAULT_SELF_HOSTED_LABELS)


def test_job_has_a_wall_clock_budget() -> None:
    """A hung job must fail instead of holding a pool runner indefinitely."""
    match = re.search(r"^\s*timeout-minutes:\s*(\d+)\s*$", _workflow_text(), re.MULTILINE)
    assert match, "job has no timeout-minutes"
    assert 1 <= int(match.group(1)) <= 180