# Agent Instructions

## Commands

**Install (development):**
```bash
pip install -e .
pip install -e ".[pt]"
pip install -e ".[tf]"
```

**Test:**
```bash
uv run pytest tests
uv run pytest tests/test_publish.py
uv run pytest -q
```

When optional dependencies are needed for focused checks, install them explicitly with `uv run --with ...`, for example:

```bash
uv run --with pytest --with pandas --with tabulate --with loguru --with requests --with py-cpuinfo pytest -q
```

For PyTorch-backed benchmark work, include the PyTorch extra or install `torch` in the test environment:

```bash
uv run --with pytest --with pandas --with tabulate --with loguru --with requests --with py-cpuinfo --with torch pytest -q
```

**Static check:**
```bash
python3 -m compileall simple_ai_benchmarking tests
```

## Unit Tests

When creating a PR, review whether the existing test suite sufficiently covers the PR's additions and behavior changes. Before merging, repeat that review if additional commits changed code since PR creation.

If the change introduces new logic, edge cases, benchmark metadata behavior, publishing behavior, CLI behavior, model/workload behavior, or fixes a regression, expand the unit tests in the same PR unless there is a concrete reason not to.

Document the decision in the PR's `**Test:**` section, including either the tests added/run or why existing coverage is sufficient.

## CI Expectations

Before committing code changes, run the smallest meaningful verification set that covers the change. For broad or shared behavior changes, run:

```bash
python3 -m compileall simple_ai_benchmarking tests
uv run pytest -q
```

If the base environment is missing optional project dependencies, use `uv run --with ...` and list the exact command in the PR's `**Test:**` section.

## Branch & PR Conventions

Workflow: GitHub Flow. Branch off `main`, open a PR, squash-merge back. `main` stays deployable.

Branch names: `<type>/<short-kebab-description>`, lowercase, no more than 50 characters. Types: `feat/`, `fix/`, `refactor/`, `docs/`, `chore/`, `test/`, `ci/`, `hotfix/`.

Commit subject: plain imperative, sentence case, no trailing period, no more than 72 characters.

PR title: same rules as commit subject.

PR description:
```markdown
**Why:** <motivation>
**What:**
- <change>
- <change>
**Test:** <commands run, or why existing coverage is sufficient>
```

Merge strategy: squash and merge only.

Agent workflow: when a feature branch is ready, draft the PR title and body in the format above, show them to the user for confirmation, and only then open the PR via `gh pr create`. If `gh` is unavailable, print the exact title and body the user should paste, plus the GitHub compare URL.

## Push Rules

Commits, pushes, and PR merges always require explicit user confirmation, even in full auto mode. Never run `git commit`, `git push`, `gh pr merge`, or equivalent merge/push commands until the user has confirmed that exact action.

Before pushing:
1. Run the relevant tests and static checks.
2. Commit only intentional source, test, migration, and documentation files.
3. Leave local artifacts unstaged, such as `.DS_Store`, generated CSV/JSON result files, local SQLite databases, caches, and logs.
4. Use `git status --short` to verify what will be pushed.
5. Push the current branch with `git push` or `git push -u origin <branch>`.

After pushing, verify that the branch tracks the remote and report the commit hash and pushed branch.

## Non-Interactive Shell Commands

Always use non-interactive flags with file operations to avoid hanging on confirmation prompts.

```bash
cp -f source dest
mv -f source dest
rm -f file
rm -rf directory
cp -rf source dest
```

For other commands that may prompt:
- `scp` - use `-o BatchMode=yes`
- `ssh` - use `-o BatchMode=yes`
- `apt-get` - use `-y`
