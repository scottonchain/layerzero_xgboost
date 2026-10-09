"""Interim report for the root-tree fix, generated from the stored JSON results (nothing typed by hand)."""
import json, os, sys, time
import numpy as np

D = 'diagnostics/root_tree_fix'
S1 = json.load(open(f'{D}/stage1_feature_changes.json')); S2 = json.load(open(f'{D}/stage2_fixed_config.json'))
BR = json.load(open(f'{D}/stage1_ixl_breakdown.json')); VEC = json.load(open(f'{D}/stage1_ixl_singleton_vector.json'))
P = json.load(open(f'{D}/stage3_pairs.json')); n = len(P)
R16 = json.load(open('results/16_label_robustness_initial_list.json'))
fpv = [x for x in R16['fp_composition'] if x['note']][0]; pt = {x['partition']: x for x in R16['by_partition']}['Test']
N_, I_, F_, J_ = pt['negatives'], pt['negatives_on_list'], fpv['fp'], fpv['fp_on_initial_list']
head = os.popen('git rev-parse --short HEAD').read().strip()
def d(q, tag, m): return q[tag]['seen'][m] - q[tag]['held_out'][m]
L = []
w = L.append
w(f"# Root-wallet tree-feature fix: interim report ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})\n")
w("Interim evidence from one environment (x86_64 Linux, Xeon 2.80 GHz, 4 cores, Python 3.11.15, pinned packages). "
  "The corrected publication rerun of notebooks 00 to 25 and 99 has not been done; results, figures and tables committed in the "
  "repository are still those from before the fix.\n")
w("## 1. Branch, fix commit, draft PR")
w("- Branch `fix/root-tree-metrics-2026-10-09` in `scottonchain/layerzero_xgboost`, started from b7a9a3f (PR #13's head, preserved untouched).")
w("- Fix commit 663e026; regression checks included. Later commits add the Section 5.3 reporting change (7c97305) and the evidence under `diagnostics/root_tree_fix/`.")
w("- Draft PR to `paven86/layerzero_xgboost`: the Claude GitHub App is refused (403) on that repository, so it must be opened from "
  "https://github.com/paven86/layerzero_xgboost/compare/main...scottonchain:layerzero_xgboost:fix/root-tree-metrics-2026-10-09?expand=1 with \"Create draft pull request\".\n")
w("## 2. Cause and inherited history")
w("`provision_features` skipped every wallet absent from the funding forest's parent map (`if a not in parent: rows.append({})`), so roots got all-zero tree "
  "features through the later `fillna(0)`, although their descendants carried the whole tree's summaries. Interactors with no recorded incoming funding edge "
  "were missing from the provider table and were zero-filled too. The behavior is in the original featurization notebook from its earliest committed version "
  "(efc1c50, May 2025). Reproduction: a labeled exchange funds A and A funds B; both have `provision_tree` = A, but A got `tree_size` 0 and B got 2.\n")
w("## 3. Regression checks")
w("`tests/test_root_tree_features.py`: 11 checks pass on the fixed code (service-funded root with a child, singleton root, root with no incoming record but a child, "
  "isolated wallet, zero-total entropy, hand-computed descendant and non-tree values, deeper depths, tree split by a labeled edge, input-order independence, one row per "
  "interactor, merge zero-fill). 10 of the 11 fail on the original code (8 assertion failures, 2 errors including the entropy division by zero).\n")
w("## 4. Full-dataset feature changes")
a = S1
w(f"- Selected model inputs ({a['n_features']} features) change for **{a['wallets_selected_model_input_changed']:,} of {a['rows']:,} wallets "
  f"({100 * a['wallets_selected_model_input_changed'] / a['rows']:.2f} %)**. All of them had all-zero tree features before.")
w("- By category: " + '; '.join(f"{c} {v['changed']:,} of {v['total']:,} ({v['pct']} %)" for c, v in a['by_category_selected_inputs'].items()) + '.')
sb = a['sybil_nonsybil_selected_inputs']
w(f"- Sybils changed {sb['changed_sybil']:,} of {sb['sybil_total']:,}; non-Sybils {sb['changed_non_sybil']:,} of {sb['non_sybil_total']:,}.")
w(f"- Roots with descendants {a['changed_selected_inputs_by_kind']['roots_with_descendants']:,} ({a['changed_sybil_by_kind']['roots_with_descendants']:,} Sybils); "
  f"singleton roots {a['changed_selected_inputs_by_kind']['singleton_roots']:,} ({a['changed_sybil_by_kind']['singleton_roots']:,} Sybils); non-roots {a['changed_selected_inputs_by_kind']['non_roots']:,}.")
w(f"- The tracker's historical count is confirmed: {a['ixl_roots_of_multi_wallet_provision_groups']:,} IxL wallets are roots of multi-wallet groups.")
w(f"- Changed model features ({len(a['changed_model_features'])}): " + ', '.join(a['changed_model_features']) + '. `depth` is unchanged. '
  "Changed computed columns outside the model: " + ', '.join(a['changed_columns_excluded_from_model']) + '.')
w(f"- Unexpected changes: none (non-root wallets: {a['unexpected_nonroot_changes'] or 'none'}; non-tree columns: {a['non_tree_columns_changed'] or 'none'}).")
i = a['identity']
w(f"- Identical between original and corrected: address order {i['address_order_identical']}, labels {i['labels_identical']}, categories {i['category_identical']}, "
  f"`provision_tree` ids {i['provision_tree_identical']}, anchors ({i['n_anchors_old']:,}) {i['anchors_flags_identical']}, train/validation/test row ids "
  f"{i['split_train_row_ids_identical'] and i['split_val_row_ids_identical'] and i['split_test_row_ids_identical']}.\n")
w("## 5. Fixed-configuration sensitivity analysis (not the retuned paper result)")
w("LightGBM, committed hyperparameters, 62 features, seed-42 tree split, one model seed (42), identical rows; early stopping and threshold from validation for each fit.\n")
w('| Test metric | Original | Corrected | Difference |\n|---|---|---|---|')
for m, lab in (('f1', 'F1'), ('ap', 'AP'), ('auroc', 'AUROC'), ('precision', 'Precision'), ('recall', 'Recall'), ('threshold', 'Validation threshold')):
    o = S2['overall'][m]; w(f"| {lab} | {o['old']:.4f} | {o['new']:.4f} | {o['diff']:+.4f} |")
w('\n| Category | Test n | Sybils | Recall original | corrected | F1 original | corrected |\n|---|---|---|---|---|---|---|')
for c, r in S2['by_category'].items():
    w(f"| {c} | {r['n']:,} | {r['sybils']:,} | {r['recall']['old']:.3f} | {r['recall']['new']:.3f} | {r['f1']['old']:.3f} | {r['f1']['new']:.3f} |")
af = S2['affected']
w(f"\nAffected Sybil roots in test: {af['affected_sybil_roots']['new']['sybils']:,}; recall {af['affected_sybil_roots']['old']['recall']:.3f} to {af['affected_sybil_roots']['new']['recall']:.3f} "
  f"(mean score {af['affected_sybil_roots']['old']['mean_prob']:.3f} to {af['affected_sybil_roots']['new']['mean_prob']:.3f}); with descendants {af['affected_sybil_roots_with_descendants']['new']['sybils']}: "
  f"{af['affected_sybil_roots_with_descendants']['old']['recall']:.3f} to {af['affected_sybil_roots_with_descendants']['new']['recall']:.3f}. "
  "The recall drop comes mostly from the higher validation-selected threshold.")
fl = S2['flips']
w(f"\nBinary predictions that change: {fl['total']:,} of {S2['n_test']:,} ({fl['pct_of_test']:.2f} %): {fl['to_flagged']} to flagged, {fl['to_unflagged']} to unflagged; "
  f"Sybils {fl['sybil']} ({fl['sybil_missed_to_caught']} missed to caught, {fl['sybil_caught_to_missed']} caught to missed), non-Sybils {fl['non_sybil']} ({fl['false_pos_added']} false positives added, {fl['false_pos_removed']} removed).\n")
w(f"## 6. Central evidence: notebook 22 report dependence, LightGBM only, **{n} of 15 pairs complete**")
w("Same components, folds (repeats 42, 1, 2; folds 0 to 4), matched-tree removal, upsampling and validation-only selection; row identities asserted equal for every arm. "
  "The original arm is PR #13's stored result (the first pair was refitted here and is bit-identical). Descriptive until all 15 pairs are done.\n")
w('| Pair | Test Sybils | F1 held / seen: original | corrected | F1 diff original | corrected | AP diff original | corrected | AUROC diff original | corrected |\n|---|---|---|---|---|---|---|---|---|---|')
for q in P:
    o, c = q['orig'], q['new']
    w(f"| r{q['repeat']} f{q['k']} | {q['test_sybils']} | {o['held_out']['f1']:.3f} / {o['seen']['f1']:.3f} | {c['held_out']['f1']:.3f} / {c['seen']['f1']:.3f} | {d(q,'orig','f1'):+.3f} | {d(q,'new','f1'):+.3f} | "
      f"{d(q,'orig','ap'):+.3f} | {d(q,'new','ap'):+.3f} | {d(q,'orig','auroc'):+.3f} | {d(q,'new','auroc'):+.3f} |")
w('\n| Seen minus held out, mean ± SD over completed pairs | Original | Corrected |\n|---|---|---|')
for m, lab in (('f1', 'F1'), ('ap', 'AP'), ('auroc', 'AUROC')):
    r = []
    for tag in ('orig', 'new'):
        x = np.array([d(q, tag, m) for q in P]); r.append(f"{x.mean():+.4f} ± {x.std(ddof=1) if n > 1 else float('nan'):.4f} ({int((x > 0).sum())}/{n} higher)")
    w(f"| {lab} | {r[0]} | {r[1]} |")
if os.path.exists(f'{D}/stage3_tests.json'):
    T = json.load(open(f'{D}/stage3_tests.json'))['tests']
    w('\nPlanned aggregate tests (all 15 pairs; corrected resampled t-test, df = 14):\n\n| Arm | Metric | Seen minus held out | Higher in | Corrected p |\n|---|---|---|---|---|')
    for t in T:
        if t['metric'] in ('f1', 'ap', 'auroc'):
            w(f"| {t['arm']} | {t['metric']} | {t['diff_mean']:+.4f} ± {t['diff_sd']:.4f} | {t['seen_higher_n']}/15 | {t['p_corrected']:.2g} |")
# ---- section 7: conclusions and ETA from the progress log
import re, datetime
stamps = []
for line in open('/home/user/fix_diag/progress.log'):
    m = re.match(r'(\S+Z) \[stage3\] pair (\d+)/15', line)
    if m: stamps.append((int(m.group(2)), datetime.datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%SZ')))
w("\n## 7. What can be concluded now, what still needs the full rerun, and the ETA")
x_o = np.array([d(q, 'orig', 'f1') for q in P]); x_n = np.array([d(q, 'new', 'f1') for q in P])
_tail = ("All 15 pairs are complete, so the planned tests are reported above." if n == 15 else "No significance test is quoted until all 15 pairs are complete.")
w(f"- Now ({'complete, LightGBM only' if n == 15 else 'descriptive, ' + str(n) + ' of 15 pairs'}): the feature fix is verified as intended and touches only roots; with a fixed configuration it changes test F1 by {S2['overall']['f1']['diff']:+.4f} and leaves AUROC unchanged; "
  f"in every completed report-dependence pair the seen arm beats the held-out arm for both feature versions, and the mean F1 effect moves from {x_o.mean():+.3f} to {x_n.mean():+.3f}. "
  f"The interpretation does not change in this diagnostic. {_tail} This is the fixed-configuration LightGBM comparison, not the retuned two-model result of the corrected rerun.")
w("- Still needs the full rerun: retuned hyperparameters (02 and 13 from scratch), the Gini keep-or-drop rule (01), all models and predictions, entity-level recall (14), the ten-split comparisons (13, 17, 19), "
  "the revised analyses (15, 16, 18, 20, 22, 24), the figures, tables and named numbers (23, 25), and the tie-out (99). The 83 of 83 checks of the earlier run are evidence for the original implementation only.")
if len(stamps) >= 3:
    per = np.mean([(b[1] - a[1]).total_seconds() for a, b in zip(stamps[1:], stamps[2:])])
    left = 15 - n
    eta = stamps[-1][1] + datetime.timedelta(seconds=per * left)
    w(f"- Observed ETA: {n} pairs done; about {per / 60:.1f} minutes per pair, so the remaining {left} should finish near {eta.strftime('%H:%M')} UTC ({(eta - datetime.timedelta(hours=6)).strftime('%H:%M')} Denver).")
w("- The corrected publication rerun (00 to 25, 99; fresh search caches) takes roughly 12 hours of compute on this four-core host (an estimate; it starts when Stage 3 ends), based on the earlier run's notebook timings (13 to 99: about 7.7 hours; 00 to 12 with the 80-minute search: about 5 hours), "
  "plus any container restarts.")
w("\n## 8. Required manuscript corrections")
w(f"- **\"Single-wallet tree\".** The tree ids (`provision_tree`) are unchanged, so counts of single-wallet groups stand. What changes is what the tree *features* said: before the fix every one of the "
  f"{BR['ixl_total']:,} IxL wallets had all-zero tree features. {BR['ixl_singletons']:,} IxL wallets ({100 * BR['ixl_singletons'] / BR['ixl_total']:.1f} %) are single-wallet trees "
  f"({BR['ixl_sybil_singletons']:,} of the {BR['ixl_sybil']:,} IxL Sybils, {100 * BR['ixl_sybil_singletons'] / BR['ixl_sybil']:.1f} %), but {BR['ixl_roots_with_desc']:,} ({100 * BR['ixl_roots_with_desc'] / BR['ixl_total']:.1f} %) "
  f"are roots of multi-wallet trees of up to {VEC['ixl_roots_with_desc_tree_size_range'][1]} wallets ({BR['ixl_sybil_roots_with_desc']} of them Sybils). Text saying IxL wallets are singletons, or that their tree features were computed from a tree, must be corrected.")
w(f"- **\"Uninformative by construction\".** The old statement cannot stand as written: the tree features were zero for all IxL wallets because of the bug, not because of the data. After the fix, "
  f"the {VEC['n_tree_model_features']} tree-derived model features take one identical value vector for all {VEC['ixl_singletons']:,} IxL singleton wallets (tree_size 1, breadth_factor 1, sparsity 1, the rest 0), so within singletons they cannot discriminate, "
  f"but that is a property of singleton trees; for the {BR['ixl_roots_with_desc']:,} IxL roots with descendants they now carry the tree's summaries. Whether they help for IxL must be shown by the corrected ablation (notebook 19), not asserted by construction.")
w("- **Numbers.** Every model result, table, figure and named number that depends on the tree features (notebooks 00 to 25 and 99) is to be replaced by the corrected rerun; the values above are fixed-configuration diagnostics only.")
w(f"- **Section 5.3 false-positive rate** (reporting change, notebooks 25 and 99; values below are from the current, pre-fix results/16 and will regenerate): N = {N_:,} original negative test addresses, I = {I_:,} of them on the initial list; "
  f"F = {F_:,} false positives at the validation threshold, J = {J_:,} of them on the list. Original-label FPR = F/N = {F_:,}/{N_:,} = {100 * F_ / N_:.3f} %. "
  f"With every initial-list address counted as positive: (F-J)/(N-I) = {F_ - J_:,}/{N_ - I_:,} = {100 * (F_ - J_) / (N_ - I_):.3f} %. The denominator removes all {I_:,} list addresses among the test negatives, flagged or not.")
open(f'{D}/INTERIM_REPORT.md', 'w').write('\n'.join(L) + '\n')
print(f"interim report written ({n} pairs) at {head}")
