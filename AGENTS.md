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

Feature-branch commit subjects should be plain imperative, sentence case, no trailing period, and no more than 72 characters.

PR titles and final squash commits to `main` should use lightweight Conventional Commit format: `<type>: <imperative subject>`. Use common types such as `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, and `ci`.

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

## Versioning

There are three independent version axes. They serve different jobs — do not conflate them.

**Package version (`VERSION` in `version_and_metadata.py`) + git tag — traceability.**
Recorded on every result as `benchmark_version` (plus `benchmark_commit_id`). It is part of `payload_hash` (dedup) but **not** of any grouping hash (`profile_hash`/`config_hash`/`runner_hash`), so it never fragments comparisons — identical configs stay comparable across releases. Bump it for any user-visible behavior change (new/changed defaults, new flags, bug fixes that change output, dependency-driven behavior changes) following SemVer: patch for behavior tweaks/fixes, minor for new backwards-compatible features, major for breaking CLI/API changes. After the PR merges to `main`, tag it: `git tag -a vX.Y.Z -m "<summary>" && git push origin vX.Y.Z`.

**Spec version (`SPEC_VERSION` / `LLM_SPEC_VERSION`) — methodology and identity schema. Rare and load-bearing.**
It is hashed into the profile, so bumping it deliberately fragments results (old and new no longer group together). Bump it **only** when either:
- the *meaning of the numbers changes for an unchanged config* (e.g. thread-pool caps that alter measured throughput, a changed TTFT/token-counting definition), or
- the *profile/identity schema changes* (an identity dimension is added or removed, e.g. `serving_engine`).

Do **not** bump the spec for: changing a default value (the profile already hashes shape — prompt/generated/context/concurrency — so a new default shape is a new profile automatically), adding/removing a reported result column (changes `payload_hash` only), refactors, or fixes that don't change measured numbers. Those are covered by the package version + commit id. When you do bump, add a comment documenting why (the existing history in `benchmark_metadata.py` is the model).

**Runner id/manifest (`*_RUNNER_ID`, `build_runner_hash`) — implementation contract.**
Bump the `.vN` runner id (or `runner_manifest_version`) only when the runner implementation changes in a way that should fragment results independently of spec/config. This is rarer still.

## Push Rules

Commits, pushes, and PR merges always require explicit user confirmation, even in full auto mode. Never run `git commit`, `git push`, `gh pr merge`, or equivalent merge/push commands until the user has confirmed that exact action.

When a task or feature is complete, automatically propose the target branch, files to include, and a commit message. If the user approves the commit, create it. If the user also asks to push, push immediately after committing; otherwise ask for separate push confirmation after the commit succeeds.

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
