# Project agent instructions

## Source of truth

Before making architectural, data, model, or evaluation decisions, read `PROJECT_CONTEXT.md`.

Section 0 of `PROJECT_CONTEXT.md` is the current project direction. If a historical section or presentation brief conflicts with it, section 0 wins.

## Current direction

This is a hybrid surveillance-video detection project.

- Violence branch (required): RWF-2000, with ResNet18 + temporal average pooling as the baseline and ResNet18 + Temporal Transformer as the main model.
- Lightweight anomaly branch (required): a separate normal profile for Ped2 and Avenue, using pretrained embeddings and an anomaly score.
- A larger anomaly model (Temporal Transformer or ConvLSTM) is optional and starts only after the two base branches, evaluation, and demo are stable.

Do not silently turn the project back into Ped2/Avenue-only anomaly detection.

## Experimental integrity

- Never claim a result, metric, dataset property, or improvement that has not been measured or supported by a source.
- Keep train, validation, test, same-domain, and external-test protocols separate.
- Do not tune thresholds or hyperparameters on test labels.
- Preserve temporal frame order and clip boundaries.
- Clearly distinguish measured results, targets, estimates, and hypotheses.

## Repository safety

- Do not commit datasets, checkpoints, credentials, access tokens, or `.env` files.
- Make small, reviewable changes; avoid unrelated refactors.
- Do not start costly full-dataset training unless explicitly requested.

## Multi-agent policy

- Use subagents only when independent or parallel work materially helps.
- Prefer read-only agents for exploration and review.
- Only one write-capable agent may own a bounded code change at a time.
- The main agent owns task decomposition, final decisions, diff review, and validation.
- For small tasks, use the main agent alone unless the user explicitly requests delegation.
