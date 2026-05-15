# Full Corpus Pairwise Readout

Date: 2026-05-15

Run artifacts: `output/corpus_pairwise_full/`

Report: `output/corpus_pairwise_full/report/index.html`

Model: `buddhist-nlp/gemma-2-mitra-e`

Scope: 16 SMDG files x 22 Txt-18 files = 352 document pairs, 8,804 total sentence embeddings, 352 persisted similarity matrices.

## Reading Notes

This is an artifact-level readout, not a final research conclusion. The scores identify candidate relationships and reading priorities. Human inspection of the matched passages is still required.

The most useful next-pass view in the report is usually `unique_both` or `diverse_both`, not raw top-k. Raw top-k is good for finding the strongest local spikes, but it can be dominated by repeated matches around the same sentence or passage.

## Score Distribution

| Metric | Min | P25 | Median | P75 | Max |
|---|---:|---:|---:|---:|---:|
| max_score | 0.5054 | 0.6233 | 0.6566 | 0.6869 | 0.8238 |
| p95_score | 0.3940 | 0.4121 | 0.4201 | 0.4320 | 0.4625 |
| mean_score | 0.2916 | 0.3065 | 0.3122 | 0.3238 | 0.3534 |
| mean_best_a_to_b | 0.3868 | 0.4491 | 0.4758 | 0.5021 | 0.5467 |
| mean_best_b_to_a | 0.3998 | 0.4584 | 0.4872 | 0.5214 | 0.5668 |

Interpretation:

- `max_score` finds the strongest single sentence-level hit.
- `p95_score` is a more stable upper-tail signal. It asks whether the strong end of the matrix is broadly elevated, not just whether one cell spikes.
- `mean_best_a_to_b` and `mean_best_b_to_a` are directional coverage signals. They differ when one text is broader, shorter, or unevenly covered by the other.

## Strongest Single-Hit Candidates

These are ranked by `max_score`.

| Pair | SMDG | Txt-18 | Max | P95 | Best A->B | Best B->A |
|---|---|---|---:|---:|---:|---:|
| A008__B013 | 08-SMDG-cog-bzhag-grel.txt | LL04_byang-chub-sems-tig.-Liljenberg-p273.txt | 0.8238 | 0.4420 | 0.4399 | 0.5557 |
| A008__B012 | 08-SMDG-cog-bzhag-grel.txt | LL03_L16_bde-ba-phra-bkod_V8-p.495-498.txt | 0.8182 | 0.4363 | 0.4699 | 0.5456 |
| A008__B010 | 08-SMDG-cog-bzhag-grel.txt | LL01_L14_rtse-mo-byung-rgyal_vol8-p480-491.txt | 0.8182 | 0.4042 | 0.5043 | 0.5044 |
| A008__B011 | 08-SMDG-cog-bzhag-grel.txt | LL02_nam-mkha-rgyal-po_vol8-v475-480.txt | 0.8182 | 0.4291 | 0.4789 | 0.5288 |
| A008__B016 | 08-SMDG-cog-bzhag-grel.txt | LL07_L10_srog-gi-_khor-lo_v8.p491-494.txt | 0.8182 | 0.4088 | 0.4482 | 0.5078 |
| A008__B018 | 08-SMDG-cog-bzhag-grel.txt | LL09_spyi-gcod_vol8_p.498.txt | 0.8182 | 0.4270 | 0.4262 | 0.5463 |
| A015__B017 | 13-ngang-thag_sems-smad-sde-dgu-TTN-v3.txt | LL08_thig-le-drug-pa-NGB-vol33_256-280.txt | 0.7903 | 0.4289 | 0.4661 | 0.5485 |
| A016__B010 | 14-SMDG-thig-le-dbyings-kyi-ti-ka-v2.txt | LL01_L14_rtse-mo-byung-rgyal_vol8-p480-491.txt | 0.7897 | 0.4037 | 0.4886 | 0.5160 |

First-pass read:

- A008 dominates the strongest single-hit list. This suggests `08-SMDG-cog-bzhag-grel.txt` has several high-intensity local overlaps with late Txt-18 files.
- The top A008 rows should be inspected in `unique_both` and `diverse_both`; raw top-k may overrepresent one repeated passage.

## Broad Upper-Tail Candidates

These are ranked by `p95_score`, which is often a better first filter for broad similarity.

| Pair | SMDG | Txt-18 | P95 | Max | Best A->B | Best B->A |
|---|---|---|---:|---:|---:|---:|
| A013__B013 | 11-SMDG-sems-lung-rgyun-thag-_grel.txt | LL04_byang-chub-sems-tig.-Liljenberg-p273.txt | 0.4625 | 0.6622 | 0.4600 | 0.5366 |
| A011__B013 | 09c-SMDG-rgyun-thag-grel-147-160.txt | LL04_byang-chub-sems-tig.-Liljenberg-p273.txt | 0.4580 | 0.6516 | 0.4675 | 0.5518 |
| A011__B012 | 09c-SMDG-rgyun-thag-grel-147-160.txt | LL03_L16_bde-ba-phra-bkod_V8-p.495-498.txt | 0.4576 | 0.7190 | 0.5039 | 0.5575 |
| A013__B012 | 11-SMDG-sems-lung-rgyun-thag-_grel.txt | LL03_L16_bde-ba-phra-bkod_V8-p.495-498.txt | 0.4573 | 0.6734 | 0.5022 | 0.5230 |
| A013__B011 | 11-SMDG-sems-lung-rgyun-thag-_grel.txt | LL02_nam-mkha-rgyal-po_vol8-v475-480.txt | 0.4571 | 0.6806 | 0.5129 | 0.5142 |
| A009__B021 | 09a-SMDG-rgyun-thag-sa-gcod.-144-145docx.txt | LL12_rje-btsan-dam-pa_NGB-vol34-p63-64.txt | 0.4551 | 0.6553 | 0.4751 | 0.4717 |
| A014__B013 | 12-SMDG-lhug-par-bzhag-pa.txt | LL04_byang-chub-sems-tig.-Liljenberg-p273.txt | 0.4550 | 0.6402 | 0.4626 | 0.5255 |
| A013__B022 | 11-SMDG-sems-lung-rgyun-thag-_grel.txt | LL13-sgom-pa-don-grub-185-r-v.txt | 0.4548 | 0.7012 | 0.4748 | 0.5278 |

First-pass read:

- The `rgyun-thag` / `sems-lung-rgyun-thag` cluster is more prominent by broad upper-tail scores than by raw max score.
- A011, A013, A014 against B012/B013/B011/B022 are good first candidates for close reading because they have stronger upper-tail behavior without relying only on one extreme cell.

## Directional Coverage Patterns

By `mean_best_a_to_b`, many SMDG files choose B001 or B003, the two `rig-pa'i khu-byug` versions, as their best Txt-18 partner.

Examples:

| SMDG | Best Txt-18 by A->B | Score |
|---|---|---:|
| A001 01-SMDG-gser-lung-non-che.txt | B001 0-Older-versions/L1-rig-pa_i-khu-byug_p302-324.txt | 0.5101 |
| A005 05-SMDG-rig-pa-khu-byug-grel.txt | B001 0-Older-versions/L1-rig-pa_i-khu-byug_p302-324.txt | 0.5243 |
| A010 09b-SMDG-rgyun-thag-gzhung-145-147.txt | B001 0-Older-versions/L1-rig-pa_i-khu-byug_p302-324.txt | 0.5385 |
| A012 10-SMDG-sems-lung-rgyun-thag.txt | B001 0-Older-versions/L1-rig-pa_i-khu-byug_p302-324.txt | 0.5467 |
| A013 11-SMDG-sems-lung-rgyun-thag-_grel.txt | B001 0-Older-versions/L1-rig-pa_i-khu-byug_p302-324.txt | 0.5457 |

By `mean_best_b_to_a`, many Txt-18 files choose A005, `05-SMDG-rig-pa-khu-byug-grel.txt`, as their best SMDG partner.

Examples:

| Txt-18 | Best SMDG by B->A | Score |
|---|---|---:|
| B001 0-Older-versions/L1-rig-pa_i-khu-byug_p302-324.txt | A005 05-SMDG-rig-pa-khu-byug-grel.txt | 0.5299 |
| B005 L3-khyung-chen_V8-p448-468.txt | A005 05-SMDG-rig-pa-khu-byug-grel.txt | 0.5357 |
| B011 LL02_nam-mkha-rgyal-po_vol8-v475-480.txt | A005 05-SMDG-rig-pa-khu-byug-grel.txt | 0.5584 |
| B012 LL03_L16_bde-ba-phra-bkod_V8-p.495-498.txt | A005 05-SMDG-rig-pa-khu-byug-grel.txt | 0.5668 |
| B013 LL04_byang-chub-sems-tig.-Liljenberg-p273.txt | A005 05-SMDG-rig-pa-khu-byug-grel.txt | 0.5622 |
| B015 LL06_L15_bde-ba-rab-_byams_vol8-p.473-475-not-sure.txt | A005 05-SMDG-rig-pa-khu-byug-grel.txt | 0.5620 |

First-pass read:

- B001/B003 may be broad semantic anchors from the Txt-18 side.
- A005 may be a broad commentary-like anchor from the SMDG side, especially for shorter Txt-18 files.
- Directional asymmetry should be interpreted as coverage behavior, not as a symmetric similarity score.

## Most Asymmetric Pairs

Largest B->A advantage:

| Pair | SMDG | Txt-18 | B->A minus A->B |
|---|---|---|---:|
| A001__B019 | 01-SMDG-gser-lung-non-che.txt | LL10_yid-bzhin-nor-bu_Vol8_431.txt | 0.1613 |
| A005__B019 | 05-SMDG-rig-pa-khu-byug-grel.txt | LL10_yid-bzhin-nor-bu_Vol8_431.txt | 0.1433 |
| A005__B018 | 05-SMDG-rig-pa-khu-byug-grel.txt | LL09_spyi-gcod_vol8_p.498.txt | 0.1400 |
| A001__B018 | 01-SMDG-gser-lung-non-che.txt | LL09_spyi-gcod_vol8_p.498.txt | 0.1357 |
| A005__B013 | 05-SMDG-rig-pa-khu-byug-grel.txt | LL04_byang-chub-sems-tig.-Liljenberg-p273.txt | 0.1341 |

Largest A->B advantage:

| Pair | SMDG | Txt-18 | B->A minus A->B |
|---|---|---|---:|
| A006__B001 | 06-SMDG-cog-bzhag-sa-gcod.txt | 0-Older-versions/L1-rig-pa_i-khu-byug_p302-324.txt | -0.1131 |
| A006__B003 | 06-SMDG-cog-bzhag-sa-gcod.txt | L1-rig-pa_i-khu-byug_V9-p302-324.txt | -0.1131 |
| A006__B005 | 06-SMDG-cog-bzhag-sa-gcod.txt | L3-khyung-chen_V8-p448-468.txt | -0.1088 |
| A006__B006 | 06-SMDG-cog-bzhag-sa-gcod.txt | L3-khyung-chen_p447-468.txt | -0.1088 |
| A003__B001 | 03-SMDG-sems-phran-rig-pa-khu-byug-sa-gcod.txt | 0-Older-versions/L1-rig-pa_i-khu-byug_p302-324.txt | -0.1039 |

First-pass read:

- Positive asymmetry often means the shorter or narrower Txt-18 text finds good homes inside a broader SMDG text, while many SMDG sentences do not reciprocally match the Txt-18 text.
- Negative asymmetry often means the selected SMDG text has many sentences that find reasonably good matches in a broader or more general Txt-18 text, while the reverse coverage is weaker.

## Recommended Reading Queue

Use the interactive report with `unique_both` first, then `diverse_both`.

1. A013__B013: broad upper-tail leader.
2. A011__B013: broad upper-tail and strong B->A coverage.
3. A011__B012: broad upper-tail plus strong directional coverage.
4. A013__B011: balanced A->B and B->A, high p95.
5. A008__B013: strongest single-hit candidate; inspect to see whether it is one passage spike or broader.
6. A015__B017: high max and meaningful B->A coverage.
7. A005__B012 and A005__B013: likely useful for commentary/coverage behavior from SMDG into short Txt-18 files.

## Next Analysis Step

Build a small review table for the first 5-10 candidate pairs:

- pair id
- view mode used (`unique_both` or `diverse_both`)
- rank
- score
- A sentence index and text
- B sentence index and text
- human note: quote, paraphrase, shared topic, false positive, uncertain

That table should become the bridge from embedding evidence to research claims.
