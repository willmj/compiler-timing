# Draft upstream PR — record

## Identifier

**PR**: [torch-spyre/torch-spyre#4113](https://github.com/torch-spyre/torch-spyre/pull/4113)
**Number**: 4113
**Status**: `OPEN` — `isDraft: true` (marked draft; reviewers not
requested; no manual labels)

## Base / head

| field       | value                                             |
| ----------- | ------------------------------------------------- |
| base        | `torch-spyre/torch-spyre:main`                    |
| base SHA    | `813a2980dbd9d2e84f5006b9cde2f305e679fc71`        |
| head        | `toddllm:dedup-constant-consumer-index`           |
| head SHA    | `ce34227eb162d5a622cb3946c1dcbdce97b6766a`        |
| commit      | signed off: Todd Deshane <todd.deshane@ibm.com>   |
| mergeable   | `MERGEABLE`                                       |

At branch-creation time, upstream `main` was still exactly the
`813a298` SHA that Phase 4 prep targeted. No adaptation required.

## Changed files (3)

| type | path                                                                                     | +   | −   |
|------|------------------------------------------------------------------------------------------|----:|----:|
| A    | `tests/configs/torch_spyre_tests/inductor/test_dedup_constants_more_config.yaml`         |   5 |   0 |
| A    | `tests/inductor/test_dedup_constants_more.py`                                            | 786 |   0 |
| M    | `torch_spyre/_inductor/dedup_constants.py`                                               |  79 |  16 |

No diagnostic instrumentation, no batch-removal code, no
`compiler-timing` references, no other unrelated changes.

## Initial CI/check state

Immediately post-open (captured via `gh pr view --json
statusCheckRollup`):

| workflow                            | status       | note                                       |
| ----------------------------------- | ------------ | ------------------------------------------ |
| DCO                                 | SUCCESS      | Signed-off-by verified                     |
| linters                             | QUEUED       | Will run ruff + pymarkdown + mypy + etc.   |
| Enforce Test CI Coverage            | QUEUED       | Confirms every `test_*.py` is CI-wired     |
| Check OOT Configs                   | QUEUED       | Validates the new test config yaml         |
| tests (detect changed files)        | IN_PROGRESS  | Preflight for the test workflow            |
| upstream-pytorch-tests (detect ...) | IN_PROGRESS  | Preflight for the upstream-torch workflow  |

Pre-PR local checks (on the laptop, in the clean worktree):

- `ruff@0.14.5 check` — clean (one unused import removed before commit).
- `ruff@0.14.5 format --check` — clean (2 files formatted before commit).
- `mypy@2.1.0` — no errors attributable to
  `torch_spyre/_inductor/dedup_constants.py`. (231 pre-existing
  errors in unrelated files, unchanged.)
- `python -m py_compile` — both new/modified Python files compile.
- `yaml.safe_load` — new config yaml parses.

Pod-side validation (against a9316b381 with an equivalent code
change via a dual-import shim; documented in
`notes/dedup-phase4-upstream-prep.md`):

- 16/16 targeted tests pass, 0 skipped. Includes
  `tests/inductor/test_opspec_tiling.py::TestOpSpecTiling::test_flash`.
- Semantic-equivalence at Lq=512, Lk=1024 — **EQUIVALENT**.

## Adaptation notes

None. Upstream `main` had not advanced between Phase 4 prep and
this PR opening; both point at `813a298`. The production
`dedup_constants.py` from `upstream-prep/` (ruff-formatted) is
byte-identical to what was pod-validated except:

1. Single-line vs multi-line formatting of one function call (ruff).
2. Removed one unused `typing.Unpack` import from the tests file (ruff).
3. Removed the `try/except ImportError` `NameSwapHandler`
   dual-import shim from the tests file (this shim only existed
   to run the tests file on `a9316b381` where `NameSwapHandler`
   is in `insert_restickify`). On current main it's in
   `pass_utils` and the direct import works.

## Concerns to resolve before "ready for review"

- Wait for CI green (linters, tests, upstream-pytorch-tests). If
  anything fails, review before marking ready.
- The PR body currently says "test_flash completed in ~104 s
  cold" — that was measured on the pod (a9316b381 tree with the
  equivalent code). If a torch-spyre reviewer runs the same test
  in their environment they should see comparable pass-local
  numbers, but absolute wall-clock will vary by hardware.

## Body file

The public PR body used is
`upstream-prep/PR-body-public.md`. The
private-repo-linked version at `upstream-prep/PR-body.md` is
retained for reference but NOT the version posted.

## What to do next

1. **Do not** click "Ready for review" until CI is green.
2. **Do not** request reviewers manually; the repo's automation
   or CODEOWNERS should handle that when the PR is ready.
3. Watch for expected review touches (Inductor-side reviewers,
   maintainers of `dedup_constants.py`, tests owner).
