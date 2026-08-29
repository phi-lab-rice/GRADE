# Anonymity and security review

MobiCom’s 2026 CFP states that artifact evaluation is **best-effort
double-blind**.  Treat this as a real submission constraint:

- Use an anonymous repository/archive or a HotCRP-uploaded archive for review.
- Remove author names, emails, affiliations, personal usernames, W&B project
  links, git remotes/history, machine paths, and named cloud buckets.
- Do not contact reviewers except through HotCRP.  If hardware access is
  necessary, use anonymous accounts and the process described in the CFP.
- Do not create a public DOI release until artifact acceptance.  The CFP asks
  for a public DOI/repository after acceptance for the Available badge.

## Security blocker

The source tree contains a live-looking W&B API credential in multiple YAML
files.  It is intentionally excluded/redacted from this package.  Revoke or
rotate it now, remove it from every source and git history before public
release, and use `WANDB_MODE=offline` (or no W&B) for reviewer workflows.

## Configuration hygiene

The original configurations contain workstation and server paths.  The copied
artifact source is a reference snapshot; reviewers must use local config copies
whose paths point to `dataset_prepare/` and `checkpoints/`.  Before submission, produce
one tested, anonymous configuration per advertised workflow and remove all
absolute paths from it.

## Remaining scrub required

A filename-only scan still finds personal absolute-path literals in 14 copied
Python files: the radar-ablation dataset helpers, GRT-GRADE/Stage-2 legacy
dataloaders, `radarcam-depth/utils/eval_utils.py`, and
`evaluation/figure_11.py`.  These are mostly local test/example paths, but the
final review archive must replace them with release-relative paths or remove
the example blocks.  This preparation directory is therefore **not yet safe to
upload to HotCRP**.
