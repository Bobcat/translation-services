# Image-translation review findings — 2026-07-04

Reviewer: Claude (Fable 5), review 2026-07-04, fixes 2026-07-04 – 2026-07-05.

Full-codebase review of the translate_image pipeline and its surroundings
(runtime/API, tasks, OCR, grouping, translation, replacement, regression
harness). Six parallel review passes, one per subsystem; the highest-severity
findings were re-verified by hand against the code. Line numbers refer to the
tree at commit `82dae53`.

**Status legend** — *confirmed*: re-verified by hand in the code (or, where
noted, numerically). *reported*: found and argued by a review pass, not
independently re-checked. A confirmed finding states a real code path; whether
it fires in practice still depends on inputs.

## Priority decision

The agreed priority is **the rendered end result of a translate_image run on
the current happy path** (defaults: `use_geometry_columns=true`,
`preserve_heuristic_text=true`, `preserve_unchanged_text=false`). Loud
failures (translategemma returning an untranslated image, HTTP 500s) are
noticed in normal use and rank below silent output-quality defects.

A second first-class track (user, same day): **concurrency under multiple
simultaneous users** — no blocking on work that can run off the event loop,
and no cross-user serialization beyond locks that are genuinely required
(a per-engine OCR lock is acceptable; a global one is not). See the
Concurrency section (C1–C4).

Fix order, most upstream first:

1. **Grouping binders** — G1, then G3, then G4. Wrong cell↔hint binding
   cascades into translation and render. G1/G3 are the prime suspects for the
   open zh align regression.
2. **Concurrency / blocking** — C1–C4 (parallel track, independent files).
3. **Parser backstops** — P1–P5.
4. **Render systematics** — R1–R4 (R5 is a behaviour choice, decide
   separately).
5. **Translation-path leaks** — T2–T5.
6. **OCR read quality** — O1–O4.
7. **Parked** — retranslate contract (RT*), regression-harness defects (RG*),
   remaining runtime/API (A*). Real, but neither output quality nor
   concurrency.

Validation per fix: `tests/` plus a full regression replay (`scripts/regress.py`),
Latin fixtures as the control group; re-accept only intended zh changes.

**Progress (2026-07-04):** G1 fixed (distinct-token full test; harness 38/39,
the one failure — `menukaart/nl/v1`, 44 diffs — is pre-existing and identical
with and without the change). C1–C3 fixed (canonicalization and testset copy
via `anyio.to_thread`; `prune()` returns stale dirs, deletion off-loop outside
the lock; per-engine OCR locks with double-checked construction) — verified by
a live end-to-end request plus one zh and one en fixture replay. G3 fixed
(below-threshold indexed match falls back to the fuzzy full scan; harness
40/40). P1–P5 fixed (level word map + pipe-gated first-letter fallback;
standalone-bare gate; bullet-marker length/`@` gate; whitespace-gated
markdown strip; docstrings corrected — harness 40/40, five new pinning
tests). `menukaart/nl/v1` was re-baselined by the user. G4 fixed (claim dedup
counts fuzzy-covered line tokens via the extracted `_token_pair_matches`;
harness 40/40, two direct `_resolve_claim_clusters` tests — its first).
R1–R4 fixed (Unicode `_field_key` with diacritic fold; x de-meaned per
cluster in `_baseline_angle`; `fold_lone_fullwidth_punctuation` before
render; single CJK char renders). Harness 38/40 — both fails verified as the
INTENDED improvements, awaiting user re-baseline: `adv-budgets/zh/v1` (R4:
leftover "iar"→"商" now renders instead of leaving the original) and
`afstand-houden/el/v2` (R2: correct steeper angle shifts the bottom block
slightly; render visually verified good); both were re-baselined by the user.
C4 fixed (per-stage cancel checkpoints via a `checkpoint` callable +
`PipelineCancelled`; a post-cancel pipeline error no longer masks the cancel).
T2/T3 fixed (the resolved prompt reaches `_translate_one`), T4 (a failed
batch-fallback call degrades that one unit — route `*_batch_fallback_failed` —
and failed calls are logged into `llm_calls`), T5 (`_parse_blocks` keeps a
leading minus; only `#`-bearing rules break blocks), T6 (a numbered reply must
number exactly 1..n or it is rejected), plus the preamble typo. O1 fixed
(upright rescue only for reads ≤ 2 chars WITHOUT CJK glyphs — content-based,
not engine-based: a language gate broke the digit rescue on the zh receipt
re-OCR), O2 (predicate mirrors PaddleX's int truncation), O3 (empty upright
read cannot delete a segment), O4 (halfwidth katakana + Ext A route to the
server pair), plus a crash guard around the upright crop. Final harness 40/40.

R5 (2026-07-05): made a per-request enum instead of a hard switch —
`render_size_mode: "min" | "median"` (default `min`, strict no-op; unknown
values 400 at the schema edge). Threads request → task → render
(`_group_size`); retranslate inherits the source run's mode (body may
override); capture/replay pin it in `request_flags` so a median-approved
fixture replays under median. Selectable in the workbench next to the
preserve/geometry options. A future smarter policy can land as another enum
value (e.g. `auto`). Harness 40/40 (fixtures default to min), 117 tests.

RG1 fixed (2026-07-05): replay compares the POST-preserve-filter units, as
capture freezes them — identity for flag-on fixtures (harness 42/42), and a
flag-off fixture is no longer born-failing. Flag/replay policy documented at
`Fixture.preserve_heuristic_text`: re-apply at replay iff the flag acts on
the re-run side of the freeze boundary (preserve_heuristic_text,
render_size_mode); translation-side flags (preserve_unchanged_text,
use_geometry_columns) are provenance-only — their effect is baked into the
frozen translations.

T1 fixed (2026-07-05): hinted units the batch left untranslated (wobbled
block, rejected numbered reply, failed translategemma window) now fall back
PER HINT LINE — one `_translate_one` call per clean VLM line (cached: units
sharing a line share the call), mapped via the extracted `_mapped_hint_line`
so structured and fallback map identically; route `*_hint_line_fallback`.
Never the unit's OCR fragment (isolated "WORKS IF" is untranslatable under
any prompt). A failed line call degrades to the old `skipped_hinted_missing`
(original pixels stay) and is logged. Four tests; harness 45/45 (frozen
translations → no-op; testset grew to 45). Follow-up (user observation): the
numbered batch is now reserved for the NO-HINT case — with hints present it
would shadow the better per-hint-line fallback with fragment-quality
translations; with no usable hint its one call is still better than N
per-unit calls.

RT1-3 fixed (2026-07-05): retranslate feeds the hint variant the source run
fed (adjusted when `use_geometry_columns`, raw fallback for old
grouping.json); `submit_retranslate` carries the source run's
preserve/geometry flags with body override (`in body`, so an explicit false
survives); the "NOT yet fed to translation" comment corrected. The
retranslate "same units, only the prompt varies" contract now holds. RG2 is
largely closed by the T1 redesign: the numbered batch only runs for no-hint
documents (leftover-keyed captures) and the hint-line fallback gives shared
lines identical text — only `preserve_unchanged_text` can still differ
per-unit on a shared line.

A/O sweep (2026-07-05): A1 (testset `name` gets the sibling resolve/
relative_to guard), A2 (unique upload filename per submission + unlink on
reject/dedupe — a conflicting resubmission can no longer replace a record's
stored input), A3 (`request_id` schema-constrained to safe_token's alphabet,
≤120 chars — kills the work-dir collision class), A4 (per-process
`instance_id` in the completions envelope for cursor resets), A5 (32 MB
upload cap → 413; `DecompressionBombError` → 400; startup sweep of orphaned
work dirs — records are in-memory, so post-restart dirs are unreachable),
A7 (prompt PUT 404s on unknown ids and hand-written meta.toml keys survive
the rewrite; failures log a traceback and the error message names the
exception class), O5-rest (missing rec_scores fail safe at 0.0), O6 (config
comment corrected). Seven new tests; live-verified sweep + instance_id.

Client-visible set resolved (2026-07-05) after the consumer inventory (two
clients: the workbench, and the asr camera app's image_translation_bridge —
neither reads events' `response` or `segments`): completion events are now
notifications (`response: null` — poll the request for the payload; kills
the up-to-1000-full-responses memory pin), retranslate emits the same
cell-shaped `segments` as translate ({id, text, bbox} from the cached
members), and framework validation errors (422s) now speak the service's
`{code, message, retryable}` dialect at 400 — one error parser for clients.

Final block (2026-07-05): T8 — the pool default was verified safe
(`allow_remote: bool = False` in llm-pool's schema); every translation
payload now pins it off anyway via the new `_post_responses()`, the one door
to `/v1/responses` that replaced the three duplicated post/raise/log blocks
(kept on module-level `httpx.post`: it is the test seam). T7 — the pool
envelope has no finish_reason; `metrics.engine_output_tokens` reaching the
decoding cap is the truncation signal (`_reply_truncated`): a truncated
structured/numbered reply is rejected (per-hint-line fallback covers it), a
truncated per-line reply degrades to empty. RG3 — the capture CLI refuses a
cell-less (retranslate) response with a pointer to the grafting endpoint,
and reports a duplicate instead of crashing on it.

Everything from the review is done. The last item, the `merging.py` removal
proposal (O6, production-dead), was approved and executed on 2026-07-19.
User-side: the min/median A/B evaluation and the zh verification.

## Grouping — alignment (app/grouping/align.py)

**G1 (high, confirmed)** — align.py:340 vs :348. The match mass sums
`_token_score` over the cell token **list** (duplicates and fuzzy 0.9s
included) but the `full` test compares that mass against the hint line's
token **set** size. A cell with repeated tokens can count as a "full match"
for a line it only half covers. Scenario: cell `小心小心` → tokens
`[小,心,小,心]`, mass 4, vs hint `小心地滑` (set size 4, `地`/`滑` unmatched)
→ marked full. Full-match wins before sticky in `_pick_hint` (:392-394), so a
wrapped continuation jumps to a wrong neighbouring line. Per-character CJK
tokens repeat constantly, so zh is hit structurally harder than Latin.
Fix shape: count distinct covered hint tokens for the `full` test only;
`score` stays as is.

**G3 (medium, reported)** — align.py:309-319. `_candidate_hints` falls back
to the full scan (where fuzzy matching can bind) only when the cell shares
**zero** exact tokens with any line. One clean token restricts scoring to
that token's lines. Scenario: hints `["totaal pas", "Kaarthouder betaling"]`,
cell `"Kaarthuder betallng pas"` → candidates `{0}` via "pas", score 0.33 <
0.4 → leftover; a full scan binds line 1 at 0.6. The comment's "verified
across the testset" is empirical, not structural; the pinning test only
covers all-clean or all-garble cells. Fix shape: also full-scan when the
indexed best score is below `_MATCH_THRESHOLD`.

**G4 (medium, reported)** — align.py:185-207. Claim dedup intersects
**exact** tokens only, while binding is fuzzy. A group that bound its line
purely via fuzzy matches has an empty exact intersection and lands in
`dropped` unconditionally — even a genuine wrapped continuation. Scenario: a
garbled continuation cell (`kortlng`) is dropped to `ignored`; its original
pixels stay rendered next to the translated line — the leftover-doubling
class the dedup exists to prevent. Fix shape: let `tokens_of` count
fuzzy-covered line tokens, or treat empty-exact claims as merge-eligible
rather than auto-redundant.

**G5 (medium-low, reported)** — heuristics.py:63,68 via align.py:277.
`_near`'s gap tolerance scales with the merged **cluster** height, so a
3-line wrapped cluster accepts a stray up to ~1.5× its own (tall) height
below it. Fix shape: use the median member line height as the gap unit in
the merge path.

**G6 (low, reported)** — align.py:369-376. A unique full-alpha match
survives the position guard only if some other candidate is nearby
(`and guarded`); with no near competitor the cell becomes a leftover
(two-column receipts). If intentional, needs a comment.

**G7 (low, reported)** — heuristics.py:36. `_is_nontranslatable` matches a
URL suffix anywhere in the cell, so `"Meer info op voorbeeld.nl"` is left
untranslated; the docstring says whole-cell URLs only. Fix shape: apply the
suffix test to single-token cells only.

**G8 (info, reported)** — tokens.py:48. Cyrillic/Greek/Arabic/Hebrew produce
zero tokens: the whole hint is inert and every cell becomes a per-cell
leftover. Known, deliberate; degradation is total, not partial (parked
"hard languages" plan).

## Grouping — hint parser (app/grouping/hint_parser.py)

**P1 (high, reported)** — :60-68 with :224. The `st` alternative needs no
pipe/colon, and `_level_of` falls back to the first letter. An unlabeled
bold line `**Menu**` parses as a standalone label (`m` → footer): the
heading is deleted from the hint units and the following lines inherit
level footer. Same for any t/h/b/m-initial bold word (`**Totaal: 58,51**` →
unit `58,51` at level title, "Totaal" deleted). The pinned test
(`**Voorgerechten:**`) survives only because "v" is not a level code.
Fix shape: for `st`/`si`, require a `|` (or an exact level word/code) before
treating the match as a label.

**P2 (medium, reported)** — :64. A text row whose first field is a level
word/letter + `|` (`Title | Mr`, receipt tax row `B | 1,69`) matches the
`bare` alternative with empty rest → phantom standalone label; the row's
text vanishes and following lines shift level/block. Fix shape: accept a
standalone `bare` label only with a `<digits>pt` field or ≥ 2 pipes.

**P3 (medium-low, reported)** — :84,96-104. `_bullet_of`'s optional marker
group greedily eats the first `|` field of a colon-less bullet item:
`…|l|@blt|Prijs | 12,50` → marker "Prijs", text "12,50". Fix shape: accept
the marker only when short (≤ ~3 chars) or `@`-prefixed.

**P4 (medium-low, reported)** — :64,107,217-225. `footer|…` parses as a
label but `_LEVEL_BY_CODE` has no `f`: the label is stripped, the level is
lost. One-line fix in the level map.

**P5 (low, reported)** — :190. `line.lstrip("-*#")` strips a genuine
leading minus: `-2,00 korting…` → `2,00 korting…` in the text the structured
translation re-translates. Sign loss on receipts. Same defect exists
independently in the translation parser (T5). Fix shape: strip `-` only when
followed by whitespace.

**P6 (low, reported)** — :7 and the `parse_grouping_output` docstring claim
`[Label]` forms parse; no alternative handles brackets. Fix the docs or
restore the tolerance.

## Translation (app/translation/translate.py)

**T1 (high, confirmed)** — :148-158 with :176-187. translategemma mode: when
`_translate_structured` returns empty (block-count mismatch — one extra
model line or `###` suffices), the numbered fallback is skipped
(`!= "translategemma"`) and the result loop marks every unit with a
`hint_index` as `skipped_hinted_missing` **before** `_translate_one` is
reached. The per-unit fallback promised by the comment at :148 is unreachable
for hinted units. The request "succeeds" with empty translations for
essentially the whole document. Loud in practice (image comes back
untranslated) but zero test coverage on this path.

**T2 (medium, reported)** — :128, :188. `batched = len(translatable) > 1`
gates the structured path: a single-unit image goes straight to
`_translate_one`, which uses the hardcoded `_system_prompt` — the resolved
prompt (custom prompt / prompt id) and the hint line are silently ignored.
A prompt A/B on a one-line sign is a no-op.

**T3 (medium, reported)** — :188-198 → :318. The per-unit batch fallback
also ignores the resolved prompt. A unit whose block wobbled is translated
under a different prompt than the rest of the document — reintroducing,
sporadically, the repeat-signs bug the flat prompt fixed.

**T4 (medium, reported)** — :324-330, :334. One failed fallback HTTP call
fails the whole request (no try around the per-unit calls; all completed
batch work discarded), and the failed call is never appended to `call_log`,
so `llm_calls/` does not show what died. No retries; fresh connection per
call.

**T5 (low, reported)** — :551-556. `_parse_blocks` strips a leading `-`
(`-25% KORTING` → sign lost) and treats a `---`-only line as a block break,
shifting counts. Counterpart of P5.

**T6 (low, reported)** — :258-261. `_parse_numbered` accepts a 0-based
reply: item 0 is dropped and every translation shifts one unit. A
`len(parsed) == len(items)` check would reject the reply.

**T7 (low, reported)** — :240, :409. `max_tokens: 4096` with no
finish-reason/truncation check at any call site: a dense image loses its
tail blocks → count mismatch → the numbered retry truncates again → tail
units end `skipped_hinted_missing`.

**T8 (medium, unverified)** — :235-241, :299-310, :416-438. None of the
three translation payloads set `allow_remote: false`; the grouping VLM call
does (vlm.py:169). If the pool default is permissive, document **text**
(receipts carry real PII) can route to a remote model while the image never
does. Verify the pool default; if permissive, this is high.

**T9 (info, reported)** — :142 uses `BUILTIN_PROMPTS[IMAGE_DEFAULT_ID]`
directly, bypassing a tuned disk override (currently byte-identical, so
latent). Typo "catgory" in the translategemma preamble (:367).

## Replacement / render (app/replacement/)

**R1 (medium, confirmed numerically)** — render.py:315,324.
`_split_table_row` normalizes with `[^a-z0-9]`: non-Latin text normalizes to
empty, every rank is zero, the split never fires. A hinted `|` row in
Cyrillic/CJK/Arabic reflows as one joined line — the value renders behind
the label. Fix shape: Unicode `\w`, as `_reproduced_in` already does.

**R2 (medium, confirmed numerically)** — render.py:669-680.
`_baseline_angle` de-means y per cluster but not x; with clusters of
different x-extents the forced common intercept biases the slope shallow.
Measured: two parallel lines at a true 6.0° fit as 4.475°. Over a 1000 px
line the right end sits ~26 px off the erased band. Fix shape: de-mean x per
cluster (fit becomes exact for parallel lines).

**R3 (medium, confirmed numerically)** — render.py:484 + fit.py:57-66.
`_has_cjk` includes fullwidth forms (U+FF00–FFEF) and CJK punctuation
(U+3000–303F): one retained `！` makes `is_cjk_text` true
(`is_cjk_text('DANGER！') is True`), flipping the **group-wide** size ratio
0.9 → 0.72 (~20% smaller) and rerouting the font to the CJK face. Masquerades
as VLM run-variance. Fix shape: classify on Han/Kana letters only, or
NFKC-fold fullwidth punctuation in translated text.

**R4 (medium, reported)** — render.py:447. `len(translated) <= 1` drops any
1-char translation as noise. Wrong for CJK targets: "PUSH" → "推" renders
nothing; the original stays. Fix shape: skip empty only, or exempt CJK.

**R5 (low, behaviour choice)** — render.py:526. Group size is the **min**
over plane targets: one under-measured lowercase line shrinks the whole
block ~30%. Median matches the one-size-per-group intent better. Decide
separately (changes many renders).

**R6 (low, reported)** — render.py:620-624. `_bullet_geometry` uses
rotated-frame coordinates directly as image-space crop bounds (only valid at
angle 0) inside a ±3° gate, and the crop is not clamped to the image (PIL
pads black = fake ink at edges). Fix shape: run only at effective angle 0,
or map corners through `geo.to_image`.

**R7 (low, reported)** — render.py:776 vs :492. `_fit_group` clamps to
160 pt but the plane target is uncapped: on a high-res photo (250 px line
height) text renders ~30% under source size, plus a wasted 225→160 shrink
loop. Cap the plan target with the same constant.

**R8 (minor, reported)** — non-premultiplied alpha through LANCZOS/warp
(dark fringe on light text, :574, :901-906); RTL targets anchor left
(hint_parser.py:230-236 maps `r`→None); `" ".join` inserts an ASCII space
between CJK units mid-sentence (:519); tile width `int()` truncation can
clip the last glyph's AA edge (:570).

## OCR (app/ocr/)

**O1 (medium-high, reported)** — paddleocr.py:107-113. The upright
re-recognition pass fires on every tall crop (h/w ≥ 1.5), including genuine
vertical CJK columns where the 90° rotation is correct; on a `>=` score tie
the upright misread replaces correct text. Fix shape: skip when the routed
language is the server pair, or when the rotated text is longer than 2 chars.

**O2 (medium, reported)** — paddleocr.py:124-138. The rotation predicate
uses float edge lengths of the raw quad; PaddleX uses int-truncated sides of
the int32 minAreaRect (verified against the venv sources). In the ~1.40–1.60
band PaddleX rotates but the rescue does not fire — the original
isolated-digit bug can resurface there. Fix shape: mirror PaddleX's int
truncation.

**O3 (low-medium, reported)** — paddleocr.py:112-115. An upright result with
**empty text** and a `>=` score deletes a valid segment. One-token fix:
require `upright[0]`.

**O4 (medium-low, reported)** — paddleocr.py:62-63. `_count_cjk_glyphs`
misses halfwidth katakana (U+FF66–FF9F) and CJK Ext A: a halfwidth-katakana
receipt hint routes to en_mobile and the whole image is misrecognized.

**O5 (low, reported)** — engine construction (minutes on first server-pair
download) holds the same global lock as all prediction (paddleocr.py:42,77,186);
missing `rec_scores` default to confidence 1.0 — fails open (:96-98); an
unguarded ~1px-wide tall crop in the upright pass can fail the whole request
(:141-155).

**O6 (cleanup, reported)** — app/ocr/merging.py is production-dead: only
`run_paddleocr(merge_lines=True)` reaches it and nothing in `app/` calls
that (`run_raw_ocr` hardwires `merge_lines=False`); only tests exercise it.
It also contains a row-absorption cascade that would be high-severity if
live. Remove or mark test-only. Config comment at config.py:46-49
contradicts the pinned-language override at paddleocr.py:171-172 (comment is
wrong, behaviour is intended).

## Concurrency / blocking (C)

Context: `runner_slots = 2` (config/settings.json); the pipeline itself runs
via `asyncio.to_thread`, so pipeline stages do not block the event loop.
What does hurt simultaneous users:

**C1 (high, confirmed)** — main.py:136. `_canonical_image_bytes` (PIL decode
+ full re-encode of the upload) runs directly in the async route handler, on
the event loop. Every concurrent user's polls/submits/artifact fetches stall
for the duration (100s of ms for a large phone photo). Same pattern:
the testset copy at main.py:251 (`read_bytes`/`write_bytes` in the handler).
Fix shape: `anyio.to_thread.run_sync`, as the regression endpoints already do.

**C2 (high, confirmed)** — records.py:111-125 with service.py:102-103,144-145.
`prune()` runs synchronous `shutil.rmtree` per expired record, and it is
called under the global `asyncio.Lock` from every submit and every
get_request. A TTL boundary (many records with debug PNGs expiring at once)
stalls the event loop **and** serializes all runtime operations — including
runner state transitions — behind the lock. Fix shape: collect removable
paths under the lock, delete after release in a worker thread.

**C3 (high, confirmed)** — paddleocr.py:42,77,186. One module-global
`threading.Lock` serializes (a) all OCR prediction across runner slots —
`run_paddleocr` holds it for predict + parse + the upright re-recognition
loop — and (b) engine construction: the first CJK request builds the server
det/rec pair (model download/compile, seconds to minutes) while a warm
Latin request of another user waits on the same lock. Fix shape: per-engine
locks with a double-checked cache insert. A per-engine lock stays (Paddle
predictors are not assumed thread-safe); only the cross-engine and
construction coupling goes.

**C4 (medium, confirmed)** — service.py:167-173, 272-296. Cancel is
advisory-only: a cancelled request keeps its runner slot for the full
OCR → VLM → translate → render run; with 2 slots, one cancelled job halves
service capacity for minutes. Fix shape: cheap record-state checkpoints
between stages (at minimum before translation), releasing the slot early.

**C5 (design note, not a bug)** — VLM and translation HTTP waits run inside
the pipeline thread, occupying a runner slot while blocked on the network.
That is the bounded-concurrency design. If multi-user throughput becomes a
goal, the lever is more slots or stage-level async — a separate decision,
not part of this track.

## Parked — retranslate contract (RT)

**RT1 (high, confirmed)** — retranslate_image.py:50 reads raw `hint_units`;
the live run feeds `hint_units_adjusted` by default
(translate_image.py:145-148). **RT2 (confirmed)** — service.py:326-338 drops
`preserve_heuristic_text` / `preserve_unchanged_text` / `use_geometry_columns`
from the retranslate payload; schema defaults apply instead of the source
run's values. **RT3 (confirmed)** — the comment at translate_image.py:110-111
("NOT yet fed to translation") is stale and contradicts :145-148. Net: a
prompt A/B via retranslate silently varies three inputs besides the prompt.
Fix before the next round of prompt work.

## Parked — regression harness (RG)

**RG1 (medium, reported)** — capture freezes post-preserve-filter units
(capture.py:197-203); replay diffs pre-filter units (replay.py:33). A
fixture captured with `preserve_heuristic_text=false` fails its first replay
forever; a resnapshot then bakes the wrong baseline in. Happy-path default
(true) is unaffected. **RG2 (medium, reported)** — `hint_translations` keyed
by `hint_index` (capture.py:47-53) is lossy when units share a hint line and
translation came via the numbered fallback (per-unit texts differ; last one
wins, replay attaches it to both units). **RG3 (low, reported)** —
scripts/capture_fixture.py: KeyError on a duplicate capture; silently writes
an unreplayable fixture (cells=[], empty hint) for a retranslate request —
the HTTP endpoint guards both.

## Parked — runtime / API (A)

**A1 (high, confirmed)** — main.py:249-251. `POST /v1/regression/testset`
joins body `name` into the path unvalidated (`mkdir` + `write_bytes`):
`../`/absolute names write outside the testset root. Every sibling endpoint
has the `resolve()`/`relative_to()` check; `TESTSET_ROOT` is also
CWD-relative (capture.py:20). One-line fix.

**A2 (high, confirmed)** — main.py:155-157. The upload is written to
`_uploads/<safe_token(id)>/input.*` before `runtime.submit` runs the
dedupe/409 logic: a conflicting resubmission replaces an existing request's
stored input image, then gets rejected. A queued original runs on the wrong
pixels.

**A3 (medium, confirmed)** — util.py:23-27. Records key on the raw
`request_id`, directories on `safe_token(id)` (120 chars, charset
collapsed): distinct ids can share one work dir; TTL-pruning one rmtrees the
other's live artifacts. Fix: validate `request_id` (regex + length) at the
schema edge.

**A4 (medium, reported)** — completions cursor: per-process `seq` restarts
at 1, `next_seq = max(safe_since, …)` echoes a stale large cursor forever
(service.py:176-183,502-503) — pollers strand after every restart. Add a
per-process instance id to the envelope. The 1000-event deque also pins full
responses incl. `llm_calls` in memory past record TTL (:490-503).

**A5 (medium, reported)** — no upload size cap (main.py:127);
`DecompressionBombError` escapes `_canonical_image_bytes` (:473-476) as a
500. Rejected submits leak `_uploads/` dirs; records are in-memory only, so
after every restart all prior work dirs are orphaned — `work_root` grows
unbounded (startup sweep needed).

**A6 (medium, reported)** — blocking on the event loop: PIL re-encode in the
route handler (main.py:136); `shutil.rmtree` inside `prune()` under the
runtime lock (records.py:125) stalls the loop and serializes all runtime ops
at TTL boundaries.

**A7 (low, reported)** — cancel is advisory-only (full pipeline still runs;
a post-cancel failure overwrites the state to `failed`); `PUT /v1/prompts/{id}`
is a silent upsert (store.py:82-83) and `_write` drops `title`/`notes` from
meta.toml (only `tags` round-trips); two error-envelope dialects (FastAPI
422 vs `{code,message,retryable}`); `response["segments"]` has two schemas
(cells vs units) between translate and retranslate;
failure records keep only `str(exc)` — no exception type, no traceback, no
logging (service.py:272-288).

## Verified clean (checked deliberately)

- **Replay parity**: the regression replay calls the same parse → align →
  preserve-filter → render functions with the same arguments as the live
  path; skipping `geometry_adjusted_hints` at replay is correct (it only
  shapes the frozen translation input).
- **grouping.json** is written field-consistently by both tasks; retranslate
  chains work (modulo RT1).
- **Erase/draw ordering** in render: all sampling happens before any erase,
  all erases before all draws — overlapping units cannot erase each other's
  fresh text or sample erased pixels. `_composite` edge clipping is correct.
- **EXIF orientation** is normalized at upload, so OCR and render agree on
  pixel space.
- **PaddleX parallel arrays** (texts/scores/polys/boxes) are filtered
  consistently in 3.6 — no index skew; the event loop is safe from the
  pipeline (runs via `asyncio.to_thread`).

## Test gaps (by payoff)

- Grouping: `_resolve_claim_clusters` (keep/merge/drop/fixpoint) has no
  direct tests; the indexed-equals-full-scan test lacks the mixed
  exact+garble cell that would catch G3; no parser tests for P1/P2/P3/P4.
- Translation: nothing on the translategemma structured path (T1), the
  numbered fallback, structured→numbered sequencing, or error propagation;
  no prompt-store CRUD tests.
- Render: all fixtures are axis-aligned — nothing exercises tilt (R2, R6),
  CJK wrap/sizing, bullets, or the condense/shrink loops.
- API: no tests for cancel, queue-full, duplicate/dedupe, TTL pruning,
  completions cursor, or traversal (A1 is a one-line test).
