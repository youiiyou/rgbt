# NuanceID Project Guide

This file defines the research and engineering contract for the whole
repository. It is intentionally method-agnostic where the paper has not yet
chosen a concrete implementation.

## Fixed Research Scope

- Paper title: `NuanceID: Dynamic-Static Identity Nuance Learning for
  Video-Based Visible-Infrared Person Re-Identification`.
- The task is text-based visible-infrared video person re-identification.
  The user supplies text and the system retrieves person video tracklets.
- The baseline is IRRA. The existing CLIP image encoder must be extended to
  consume a sequence of frames and to support RGB and infrared inputs.
- The paper studies HITSZ-VCM and BUPTCampus/BUPT. The datasets live outside
  this repository under `/data/ydl/datasets` and must not be copied into Git.
- Text annotations are RGB-derived. There is no assumption that an
  independent IR caption exists. The same identity description may supervise
  both modalities, but the model should be allowed to produce
  modality-adaptive text representations.

## Retrieval Protocol

Every experiment must make the gallery protocol explicit. Evaluate the same
checkpoint in three independent runs using the same query captions and
identity split:

1. RGB gallery only.
2. IR gallery only.
3. Mixed RGB+IR gallery.

The primary task is text-to-video retrieval (the current evaluator may still
use legacy `t2i` names). The abstract also promises bidirectional retrieval
results; reverse retrieval must not be silently omitted when the protocol and
metric implementation are finalized.

For a single text query, the intended research direction is to derive
modality-conditioned text features, conceptually `text_rgb` and `text_ir`.
RGB candidates are scored with the RGB-conditioned feature and IR candidates
with the IR-conditioned feature. Mixed-gallery scoring must use candidate
modality metadata and must not use an oracle identity label.

The current baseline input is six uniformly sampled frames per tracklet. A
single-frame baseline is not required for the current research path. Future
sequence-length experiments must keep sampling and preprocessing behavior
identical across compared methods.

## What the Abstract Requires

The work is not complete when it only averages frame features. The following
research questions must be addressed experimentally:

### Dynamic-Static Nuance Learning (DSNL)

- Treat identity-related nuances as the temporal modeling units rather than
  treating a complete frame as the only temporal unit.
- Separate stable appearance evidence from temporally evolving evidence.
- Provide static-only, dynamic-only, and joint ablations so improvements are
  attributable to the proposed decomposition.
- The implementation may use any suitable temporal/token/attention mechanism;
  the name of the mechanism is not fixed by this guide.

### Cross-Modal Nuance Alignment (CMNA)

- Address RGB/IR distortion in relations among identity nuances.
- Prefer relational or structural consistency objectives over blindly forcing
  heterogeneous RGB and IR feature vectors to be identical.
- Isolate CMNA in an ablation and measure both RGB-gallery and IR-gallery
  behavior. The exact alignment loss remains an open experiment variable.

### Modality-Adaptive Text Representation

- A RGB-derived caption should support both RGB and IR retrieval.
- The model may use prompts, adapters, MoE routing, conditional modules, or
  another mechanism to obtain modality-specific text features.
- These are candidate implementations, not project requirements. Compare them
  against a shared-text baseline before selecting one for the final method.

## Required Experiment Ladder

Keep a reproducible progression instead of changing several ideas at once:

1. IRRA with six-frame input, uniform frame pooling, and shared text features.
2. Modality-adaptive text representation with all other settings fixed.
3. A temporal modeling baseline with no proposed alignment module.
4. DSNL only.
5. CMNA only.
6. Full NuanceID, followed by targeted ablations.

The exact losses, weights, optimizer settings, caption variant, and candidate
module implementations are open until experiments justify them. Every run
must save its full configuration, seed, checkpoint, sequence length, caption
mode, training modalities, gallery mode, and metric output.

## Dataset and Labeling Rules

- Keep train and test identities disjoint and preserve the dataset's intended
  identity protocol.
- Tracklet records must retain identity, camera, modality, and ordered frame
  paths. Do not discard modality metadata before evaluation.
- Captions are query-side RGB annotations. Caption parsing and future caption
  revisions must be configuration-driven rather than hard-coded into model
  logic.
- VCM currently contains RGB and IR tracklets; its current loader reuses a
  canonical RGB caption for IR training samples.
- The external BUPT data contains `RGB`, `IR`, and `FakeIR` folders. The
  baseline loader expands official `RGB/IR` training rows to real RGB and real
  IR tracklets and excludes `FakeIR` entirely. Any future FakeIR experiment
  must use a separate explicit protocol and result name.
- Do not mix real IR and `FakeIR` in the same reported result unless the
  experiment names and reports state this explicitly.
- Evaluation must not use captions or frames from the wrong split, and must
  not select frames using identity labels or test-time ground truth.

## Engineering Rules

- Preserve existing user changes in the dirty worktree. Inspect staged and
  unstaged state before editing; never reset, checkout, or clean broadly.
- Keep data, checkpoints, TensorBoard logs, and generated caches outside the
  tracked source tree when practical. Do not commit `__pycache__`, model
  weights, or experiment logs.
- Prefer the existing IRRA interfaces and configuration style. New research
  modules should be enabled by explicit options and should not silently change
  the baseline path.
- Keep tensor contracts explicit: image sequences use `[B, T, C, H, W]` at the
  model boundary, and modality is carried alongside samples wherever routing
  or evaluation needs it.
- Training and evaluation aggregation must use compatible feature definitions.
  Diagnostic pooling modes must not be presented as final methods when their
  train/test behavior differs.
- Add focused checks for dataset counts, frame ordering/sampling, modality
  routing, tensor shapes, mixed-gallery scoring, and finite losses before long
  training runs.

## Project Coordination and Worktrees

- The long-lived Codex task for this repository is the NuanceID project bus.
  It owns roadmap decisions, task dispatch, review, integration, and experiment
  comparison; it should not implement large features while child tasks are
  modifying related code.
- Create one separate task per bounded deliverable. Code-writing and long GPU
  tasks must use an isolated Git worktree created from the current clean
  NuanceID integration branch. Do not run multiple writable tasks in
  `/data/ydl/project/rgbt`.
- Tasks that modify overlapping modules are serialized. GPU training runs are
  also serialized unless the user explicitly assigns separate devices.
- A child task must report its source commit, resulting commit, changed
  behavior, tests, data root, annotation checksum, query count, and unresolved
  issues. It must not merge itself into the integration branch.
- The project bus reviews and merges completed child branches. Archive or close
  a child task after its result has been integrated or deliberately rejected.
- Experiment tasks must run from a fixed commit and must not silently consume
  source changes made after the run started.

The repository split has been completed from base commit `6154116`:

1. `codex/archive-pre-nuanceid` at `1d90b98` preserves valuable BUPT, MLM,
   text-guided-pooling, adapter, diagnostic, and historical script work.
2. `codex/nuanceid-baseline` is the clean shared-text integration baseline for
   both VCM and BUPT.

Generated bytecode must not be committed. New experiments branch from the
fixed baseline commit after its CPU/GPU validation and must not modify the
archive branch.

## VCM Caption and Query Contract

Formal Stage 1 uses a hybrid data source instead of replacing the VCM loader
with a caption-only JSON loader:

- The VCM directory tree remains authoritative for RGB/IR tracklets, ordered
  frame paths, cameras, and modality labels.
- `VCM.json` is authoritative only for RGB-derived captions and the query
  protocol. It currently contains 1485 train and 1261 test RGB-camera records.
- The loader interface must expose an explicit caption source, conceptually
  `legacy` or `json`. Missing options in historical configs resolve to
  `legacy`; all new formal Stage 1 scripts explicitly select `json`.
- In JSON mode, an RGB training tracklet uses the captions of its own
  `(split, pid, camera)` record. An IR training tracklet reuses the captions of
  the naturally sorted first RGB camera for the same identity, matching the
  existing canonical-caption behavior.
- `train_caption_mode=single` selects the first caption and `double` selects
  both captions. Evaluation uses the first caption of every test record marked
  `is_query=true`; all three galleries must reuse this exact query list.
- JSON mode must validate one-to-one RGB PID/camera coverage between the JSON
  and directory tree. Missing, duplicate, or extra records are errors rather
  than silent fallbacks.
- Every formal run snapshots the selected JSON into its output directory and
  records the original path, snapshot path, SHA256, and query count. Testing
  reloads that snapshot so later edits to the live dataset cannot change the
  evaluation denominator.
- The provisional checkpoint continues to use
  `/tmp/vcm_single_caption_root` with `legacy` mode and 1260 queries. Formal new
  training uses the current 1261-query JSON. These result sets are reported
  separately.

## Immediate Task Queue

Do not start adapter, DSNL, CMNA, MLM, FakeIR, or variable-length development
until the shared-text baseline has completed this queue:

1. Run the real GPU single-batch smoke test for VCM and BUPT from the fixed
   baseline commit.
2. Train the VCM shared-text baseline for 30 epochs and independently evaluate
   RGB, IR, and mixed galleries.
3. Train the BUPT shared-text baseline with the identical model and optimizer
   settings, then independently evaluate all three galleries.
4. Freeze the baseline result table with commit, annotation/protocol hashes,
   query counts, costs, and both retrieval directions.
5. Create one isolated branch for the modality-adaptive text experiment; only
   after that comparison is resolved, proceed through the required ladder.

## Reporting Checklist

For each method and dataset, retain:

- RGB-only, IR-only, and mixed-gallery results;
- the configured sequence lengths and sampling policy;
- text-to-gallery metrics at least R1, R5, R10, mAP, and mINP;
- reverse retrieval metrics when the bidirectional protocol is enabled;
- parameter/training-cost changes and the exact baseline configuration;
- ablations for static/dynamic components, modality-adaptive text, and
  relational alignment;
- qualitative retrieval examples and failure cases for both modalities.

## Current Repository State

- Caption tooling includes `json_vcm.sh`, `json_bupt.sh`, and the shared JSON
  builder. External annotations are VCM: 2746 records/5492 captions/1261
  queries and BUPT: 7019 records/1076 queries.
- Both formal loaders consume the JSON schema. VCM retains legacy directory
  caption compatibility only for historical checkpoint evaluation.
- VCM and BUPT share the same six-frame, shared-text, uniform-pooling,
  SDM+ID model/training/evaluation path.
- BUPT uses official train plus train_auxiliary identities, real RGB plus real
  IR tracklets, and excludes FakeIR. Its four protocol files are snapshotted
  with every formal run.
- Unit tests cover caption building, sampling, transform consistency, VCM/BUPT
  split contracts, FakeIR exclusion, finite SDM, temporal pooling, bidirectional
  metrics, and experiment snapshots.
- Real external-data checks have validated VCM as 500 train IDs/2961 train
  tracklets/1261 queries and BUPT as 2004 train IDs/9008 train
  tracklets/1076 queries; both produce `[B,6,3,384,128]` batches.
- Real CLIP CPU forward/backward smoke validation is complete for VCM and BUPT.
  Managed execution currently exposes no CUDA, so GPU smoke and 30-epoch runs
  remain outstanding and must be launched from the user's GPU terminal.
- Historical text-guided pooling, MLM, IR-only, grayscale IR, adapter, and
  diagnostic work exists only on `codex/archive-pre-nuanceid`; it is not part
  of this baseline or a validated NuanceID contribution.
- The provisional six-frame checkpoint under
  `logs/VCM/20260423_233610_vcm_6frame_1caption_lr5e6_warmup1_ep10` points to
  `/tmp/vcm_single_caption_root` and has 1260 query captions. The live
  `/data/ydl/datasets/vcm` tree currently has an additional caption at
  `Test/0919/rgb/D9/caption.txt`, so it produces 1261 queries. Do not compare
  results across these caption snapshots as if their evaluation sets were
  identical; every reported table must retain `num_queries` and the data root.

When a future implementation decision is not covered here, choose the
simplest option consistent with IRRA and record the decision in the experiment
configuration and logs.

Use the existing Conda environment with `conda run -n rgbt ...` for Python
commands. GPU training may need to be launched from the user's terminal when
CUDA is not exposed to the managed execution environment.
