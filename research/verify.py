"""Re-run every experiment and check the published numbers still come out.

Why this exists
---------------
`CLAIMS.md` says a row whose experiment does not reproduce its number is a defect to fix. That is
only true if someone checks. The shared machinery -- the corpus generators, `engine.basis` -- sits
under every experiment, and anything downstream of it can drift without a single test failing,
because the tests pin properties and the ledger pins numbers.

So this runs every experiment from scratch and compares what comes back against what is published.
It reports drift rather than asserting against it: a number that moved may mean a defect, or a
deliberate improvement whose ledger row was not updated, and only a person can tell which.

Run
---
    python research/verify.py              # everything, re-measured
    python research/verify.py exp03        # one
    python research/verify.py --paths      # resolve every registered figure against the stored
                                           # results, in seconds; catches a mis-registered path
                                           # without waiting for a full re-measurement
"""
from __future__ import annotations

import importlib
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE / "experiments"))
sys.path.insert(0, str(HERE.parent / "src"))

#: Every experiment, and the published figures to check it against. Each check is
#: `(label, path into the result, expected)`; a path is a sequence of keys and indices.
CHECKS = {
    "exp01_association_vs_scalar": [
        ("z top-1 at every facet x gain cell", ("part_b", "*", "top1", "z"), 1.000),
        ("cos at R=6 g=3.0", ("part_b", -1, "top1", "cos"), 0.0625),
        ("dot at R=6 g=3.0", ("part_b", -1, "top1", "dot"), 0.000),
        ("shared-count control on unrelated frames",
         ("part_a", "negative_control", "inclusion_exclusion_mean"), 0.075),
    ],
    "exp02_refusal": [
        ("the cut is inert", ("the_cut_is_inert",), False),
        ("cells the cut excludes", ("cells_the_cut_excludes",), 21),
        ("unsound cells the cut excludes", ("unsound_cells_the_cut_excludes",), 3),
        ("unsound cells at a cut of one", ("soundness_at_every_cut", 0, "unsound"), 3),
        ("unsound cells at a cut of twenty", ("soundness_at_every_cut", 7, "unsound"), 0),
        ("cells judged", ("cells_judged",), 15),
        ("cells where precision < 1.000", ("cells_unsound",), 0),
    ],
    "exp03_relation_types": [
        ("kinds counted at R=32, noise 0.05",
         ("calibration_disjoint", -1, "by_noise", "0.05", "mean"), 32.0),
        ("within-kind alignment at R=2", ("typing", 0, "within_kind"), 0.952),
        ("between-kind alignment at R=2", ("typing", 0, "between_kind"), -0.953),
        ("negative control", ("negative_control", "mean_modes"), 0.875),
    ],
    "exp05_forward_pass": [
        ("analogy at R=6", ("accuracy", 3, "analogy"), 0.981),
        ("operator removed at R=6", ("accuracy", 3, "no_operator"), 0.167),
        ("wrong exemplar at R=6", ("wrong_operator", 3, "wrong"), 0.002),
    ],
    "exp06_learning": [
        ("order independence: disagreements", ("order_independence", "mismatches"), 0),
        ("locality: answers changed", ("locality", "changed"), 0),
        ("new structure at +4 kinds", ("new_vs_repeated", -1, "new_structure"), 8.0),
        ("repeated structure at +4", ("new_vs_repeated", -1, "repeated_structure"), 4.0),
    ],
    "exp07_real_text": [
        ("moby: first-half PMI", ("texts", 1, "top1", "pmi_first_half"), 0.521),
        ("moby: association z", ("texts", 1, "top1", "association_z"), 0.306),
    ],
    "exp08_context": [
        ("moby: best count aggregation", ("texts", 1, "best_count"), 0.288),
        ("moby: best association read", ("texts", 1, "best_association"), 0.236),
    ],
    "exp09_axis_condition": [
        ("association off-axis", ("rows", 0, "association"), 0.115),
        ("cosine off-axis", ("rows", 0, "cosine"), 0.390),
        ("association fully on-axis", ("rows", -1, "association"), 1.000),
    ],
    "exp11_text_diagnostics": [
        ("identity removed: records resolving",
         ("identity_component", "baseline removed", "resolving"), 193),
        ("identity removed: mean rank",
         ("identity_component", "baseline removed", "mean_rank"), 0.76),
        ("raw: records resolving", ("identity_component", "raw", "resolving"), 302),
        ("raw: mean rank", ("identity_component", "raw", "mean_rank"), 1.46),
        ("segment sweep: resolving at 32", ("segment_sweep", 0, "resolving"), 302),
        ("segment sweep: resolving at 128", ("segment_sweep", -1, "resolving"), 302),
        ("mock/turtle z", ("collocations", "named", 0, "z"), 9.70),
        ("mock/turtle tightness", ("collocations", "named", 0, "tightness"), 0.994),
        ("alice/said tightness", ("collocations", "named", -1, "tightness"), 0.755),
        ("random pairs, 95th percentile", ("collocations", "random", "p95"), 4.44),
        ("alice: axis separation", ("axis_dial", 0, "axis_separation"), 0.004),
        ("alice: profile separation", ("axis_dial", 0, "profile_separation"), 0.121),
        ("moby: axis separation", ("axis_dial", 1, "axis_separation"), 0.069),
        ("second-order, moby: subspace", ("second_order", 1, "subspace"), 0.297),
        ("second-order, moby: vector cosine", ("second_order", 1, "vector_cosine"), 0.640),
    ],
    "exp12_admission_and_rejects": [
        ("admission: off-axis synthetic", ("admission", 0, "margin_z"), -1.77),
        ("admission: on-axis synthetic", ("admission", 2, "margin_z"), 35.60),
        ("admission: triple corpus", ("admission", 3, "margin_z"), 39.10),
        ("admission: alice", ("admission", 4, "margin_z"), 0.08),
        ("admission: moby", ("admission", 5, "margin_z"), 0.12),
        ("admission with the scalar statistic instead",
         ("admission", 3, "margin_z_with_scalar_statistic"), 0.07),
        ("frequency strata where the association leads", ("frequency_strata", "z_leads_in"), 0),
        ("frequency strata, total cells", ("frequency_strata", "of"), 6),
        ("transfer: moby -> moby edge", ("transfer", 0, "edge"), -0.149),
        ("transfer: moby -> alice edge", ("transfer", 1, "edge"), -0.070),
        ("absence: mean share of record energy", ("absence", "mean_share"), 0.092),
        ("absence: max share", ("absence", "max_share"), 0.895),
    ],
    "exp13_weather": [
        ("admission, raw", ("admission", 0, "margin_z"), 6.86),
        ("admission, national mean removed", ("admission", 1, "margin_z"), 3.01),
        ("raw: aggregate correlation", ("results", 0, "aggregate correlation"), 0.810),
        ("raw: association", ("results", 0, "association"), 0.713),
        ("centred: aggregate correlation", ("results", 1, "aggregate correlation"), 0.777),
        ("centred: association", ("results", 1, "association"), 0.767),
        ("mean resolved rank of a station record",
         ("results", 1, "facets", "mean_rank"), 1.00),
    ],
    "exp14_multichannel": [
        ("raw arm, edge in paired standard errors",
         ("results", 0, "edge_in_paired_standard_errors"), -2.10),
        ("centred arm, edge in paired standard errors",
         ("results", 1, "edge_in_paired_standard_errors"), -0.86),
        ("raw arm separated at two standard errors",
         ("results", 0, "separated_at_two_standard_errors"), True),
        ("centred arm separated at two standard errors",
         ("results", 1, "separated_at_two_standard_errors"), False),
        ("admission, raw", ("admission", 0, "margin_z"), 66.76),
        ("admission, city mean removed", ("admission", 1, "margin_z"), 10.99),
        ("resolved rank of a station record", ("facets", 1, "mean_rank"), 1.75),
        ("narrow-and-loud pairs", ("configuration", 1, "narrow_and_loud"), 12),
        ("centred: association", ("results", 1, "association"), 0.497),
        ("centred: max channel correlation",
         ("results", 1, "max channel correlation"), 0.511),
        ("centred: random partner", ("results", 1, "random"), 0.288),
        ("centred: oracle", ("results", 1, "oracle"), 0.625),
    ],
    "exp15_facet_independence": [
        ("cells swept", ("cells",), 112),
        ("largest edge anywhere", ("edge_max",), 1.000),
        ("worst edge anywhere", ("edge_min",), -0.917),
        ("cells where a scalar beats the association",
         ("cells_where_the_association_loses",), 27),
        ("width x noise rows where coupling reduces the edge",
         ("rows_where_coupling_reduces_the_edge",), 16),
        ("32 channels, noise 0.10, independent", ("edge_at_32ch_n010_independent",), 0.983),
        ("32 channels, noise 0.10, coupled", ("edge_at_32ch_n010_coupled",), -0.017),
        ("11 channels, noise 0.10, coupled", ("edge_at_11ch_n010_coupled",), -0.917),
    ],
    "exp18_ambient_width": [
        ("cells with any edge", ("cells_with_an_edge",), 121),
        ("cells swept", ("cells",), 405),
        ("edge at the widest record, low noise", ("edge_at_widest_low_noise",), 1.000),
        ("edge at narrow records, noise 0.10", ("edge_at_narrow_mid_noise",), 0.200),
        ("edge at narrow records, noise 0.20", ("edge_at_narrow_high_noise",), 1.000),
    ],
    "exp39_audit_context": [
        ("cells where counting wins", ("cells_where_counting_wins",), 4),
        ("cells undecided", ("cells_undecided",), 6),
        ("cells where the association leads", ("cells_where_the_association_leads",), 0),
        ("alice at the published context, in paired standard errors",
         ("cells", 2, "edge_in_paired_standard_errors"), -1.44),
        ("moby-dick at the published context, in paired standard errors",
         ("cells", 7, "edge_in_paired_standard_errors"), -1.59),
        ("the claim is overturned anywhere", ("the_claim_is_overturned_anywhere",), False),
        ("cells scored", ("cells_scored",), 10),
    ],
    "exp38_audit_window": [
        ("cells where counting wins", ("cells_where_counting_wins",), 5),
        ("cells undecided", ("cells_undecided",), 5),
        ("cells where the association leads", ("cells_where_the_association_leads",), 0),
        ("cells scored", ("cells_scored",), 10),
        ("the association's best edge", ("best_edge_for_the_association",), 0.115),
        ("alice at the published window, association", ("cells", 2, "association"), 0.125),
        ("alice at the published window, counting", ("cells", 2, "counting"), 0.125),
        ("alice at the published window, in paired standard errors",
         ("cells", 2, "edge_in_paired_standard_errors"), 0.00),
        ("moby-dick at the published window, association", ("cells", 7, "association"), 0.306),
        ("moby-dick at the published window, counting", ("cells", 7, "counting"), 0.521),
        ("moby-dick at the published window, in paired standard errors",
         ("cells", 7, "edge_in_paired_standard_errors"), -4.43),
    ],
    "exp55_a_relation_arrives": [
        ("arrivals measured", ("arrivals_measured",), 3),
        ("steps where the screen counts what has arrived",
         ("steps_where_the_count_is_right",), 3),
        ("answers already given that moved", ("answers_that_moved",), 0),
        ("the count never went backwards", ("the_count_never_went_backwards",), True),
        ("questions answered at the first step", ("answered_at_the_first_step",), 34),
        ("and at the last", ("answered_at_the_last_step",), 71),
        ("an existing relation's separation, first",
         ("drift_of_the_relations_already_present", "alpha", "first"), 7.36),
        ("and last", ("drift_of_the_relations_already_present", "alpha", "last"), 4.78),
        ("the widest an existing relation's reading moves",
         ("widest_move_of_an_existing_relation",), 4.62),
        ("it stays present throughout regardless",
         ("drift_of_the_relations_already_present", "alpha", "still_present_throughout"), True),
        ("records at the first step", ("records_at_the_first_step",), 48),
        ("and at the last", ("records_at_the_last_step",), 80),
    ],
    "exp51_road_network_corpus": [
        ("sensors", ("sensors",), 307),
        ("recorded directed road edges", ("road_edges",), 340),
        ("five-minute steps", ("steps",), 16992),
        ("cells swept", ("cells_swept",), 120),
        ("cells where the recorded road relation is present",
         ("cells_where_the_road_relation_is_present",), 0),
        ("cells where the unconnected control leaked",
         ("cells_where_the_control_leaked",), 0),
        ("cells where downstream reads above unconnected",
         ("cells_where_downstream_beats_unconnected",), 120),
        ("the strongest reading the road relation reaches",
         ("best_downstream_separation",), 0.680),
        ("the control in that same cell", ("unconnected_at_the_best_cell",), 0.150),
        ("representations tried", ("representations_tried",), 3),
        ("representations that found it", ("representations_that_found_it",), 0),
        ("the best any representation reaches", ("best_representation_downstream",), 0.945),
        ("the corpus carries its own recorded relation",
         ("the_corpus_carries_the_road_relation",), False),
    ],
    "exp47_effective_sample_size": [
        ("the factor where there is no autocorrelation, so nothing at zero moves",
         ("factor_at_no_autocorrelation",), 1.004),
        ("the level it delivers there", ("level_at_no_autocorrelation",), 0.045),
        ("corrected, at a row correlation of 0.6", ("cells", 2, "corrected"), 0.058),
        ("corrected, at 0.8", ("cells", 3, "corrected"), 0.065),
        ("corrected, at 0.9", ("cells", 4, "corrected"), 0.080),
        ("the worst corrected level, at 0.99", ("worst_corrected_level",), 0.2175),
        ("the worst level as stated, at the same place", ("worst_level_as_stated",), 0.7575),
        ("the effective-sample factor at 0.9", ("cells", 4, "mean_factor"), 6.57),
        ("it still fires on records that share everything",
         ("power_at_full_sharing",), 1.0),
        ("what it costs at weak sharing and strong autocorrelation",
         ("power", 6, "corrected"), 0.765),
        ("the wide corpus, uncorrected", ("wide_corpus_as_stated",), 0.175),
        ("the wide corpus, corrected — where the surrogate read 0.058",
         ("wide_corpus_corrected",), 0.150),
        ("cosine on the same questions, which neither correction touches",
         ("wide_corpus_cosine",), 0.233),
    ],
    "exp46_wide_corpus_corrected_null": [
        ("tasks scored", ("tasks_scored",), 4),
        ("tasks the correction moved past the level's own cutoff",
         ("tasks_where_correcting_the_null_moved_the_score",), 2),
        ("the largest move, in paired standard errors",
         ("largest_move_in_paired_standard_errors",), 3.45),
        ("raw, same activity: the association as published",
         ("tasks", 0, "scores", "association"), 0.175),
        ("raw, same activity: with the decoupled null instead",
         ("tasks", 0, "scores", "association, corrected null"), 0.058),
        ("raw, same activity: cosine, which the correction does not touch",
         ("tasks", 0, "scores", "cos"), 0.233),
        ("mean removed, same activity: the association as published",
         ("tasks", 2, "scores", "association"), 0.233),
        ("mean removed, same activity: with the decoupled null instead",
         ("tasks", 2, "scores", "association, corrected null"), 0.150),
        ("raw arm: surrogate spread for pairs sharing an activity",
         ("surrogate_spread_by_arm", "raw", "spread_for_pairs_sharing_an_activity"), 3.86),
        ("raw arm: surrogate spread for pairs that do not",
         ("surrogate_spread_by_arm", "raw", "spread_for_pairs_that_do_not"), 1.68),
        ("raw arm: how much wider for the pairs the read must find",
         ("surrogate_spread_by_arm", "raw", "ratio"), 2.30),
        ("mean-removed arm: the same ratio",
         ("surrogate_spread_by_arm", "corpus mean removed", "ratio"), 2.20),
        ("same subject, raw: the correction does not move it",
         ("tasks", 1, "association, corrected null minus association, in paired standard errors"),
         -0.33),
    ],
    "exp45_null_on_autocorrelated_rows": [
        ("the level the caller asks for", ("stated_far",), 0.05),
        ("the level delivered with no row autocorrelation",
         ("realised_far_with_no_autocorrelation",), 0.045),
        ("the worst level delivered across the sweep", ("worst_realised_far",), 0.748),
        ("the autocorrelation it is worst at", ("worst_realised_far_at_rho",), 0.99),
        ("the autocorrelation from which the level stops being delivered",
         ("affected_from_rho",), 0.3),
        ("the worst level the decoupled null delivers over the same sweep",
         ("worst_realised_far_with_the_decoupled_null",), 0.065),
        ("it still fires on records that share everything",
         ("power_kept_at_full_sharing",), 1.0),
        ("what it costs at weak sharing and strong autocorrelation",
         ("power_lost_at_weak_sharing",), 0.73),
        ("corpora read in this paper that sit in the affected regime",
         ("corpora_inside_the_affected_regime",), 6),
        ("corpora placed on the curve", ("corpora_placed",), 18),
        ("the worst level any corpus read here gets",
         ("worst_realised_far_among_the_corpora_read_here",), 0.265),
        ("at no autocorrelation the two nulls agree", ("cells", 0, "with_the_decoupled_null"),
         0.055),
    ],
    "exp44_backing_from_the_relation": [
        ("noise levels swept", ("levels", "*", "asked"), 120),
        ("the best admissible rule's worst level", ("worst_level_of_the_best_rule",), 0.950),
        ("one edge, at its worst level", ("worst_level_of_one_edge",), 0.167),
        ("every edge, at its worst level", ("worst_level_of_every_edge",), 0.667),
        ("levels where the best rule beats a single edge",
         ("levels_where_the_best_rule_beats_one_edge",), 4),
        ("a relation that is not one: answered from a single edge",
         ("control", "one edge", "answered_share"), 0.110),
        ("a relation that is not one: answered by the chosen rule",
         ("control", "largest that resolves", "answered_share"), 0.147),
        ("the chosen rule's inflation, in corpus-to-corpus spreads",
         ("control", "largest that resolves", "above_one_edge_in_spreads"), 0.25),
        ("the ruled-out retry's inflation, in the same spreads",
         ("inflation_of_the_ruled_out_retry_in_spreads",), 5.69),
        ("the ruled-out retry is admissible", ("the_ruled_out_retry_is_admissible",), False),
        ("the smallest inflation this control can see",
         ("smallest_inflation_this_control_can_see",), 0.297),
    ],
    "exp43_backing_read_not_swept": [
        ("noise levels swept", ("noise_levels",), 6),
        ("levels where 'largest that resolves' matches or beats the best fixed backing",
         ("levels_where_most_matches_or_beats_the_best_fixed",), 6),
        ("levels where 'what most agree on' matches or beats it",
         ("levels_where_agreed_matches_or_beats_the_best_fixed",), 6),
        ("worst shortfall of 'largest that resolves'", ("worst_shortfall_of_most",), 0.010),
        ("lowest accuracy of 'largest that resolves'", ("lowest_accuracy_of_most",), 0.980),
        ("a relation that is not one: answered from a single edge",
         ("hollow_answered_from_one_edge",), 0.160),
        ("a relation that is not one: answered by 'largest that resolves'",
         ("hollow_answered_by_most",), 0.670),
        ("a relation that is not one: answered by 'what most agree on'",
         ("hollow_answered_by_agreed",), 0.650),
        ("either rule is admissible", ("either_rule_is_admissible",), False),
        ("a real relation, answered from a single edge",
         ("control", "alpha", "one_edge_share"), 0.820),
    ],
    "exp42_relation_vertex": [
        ("noise levels swept", ("noise_levels_swept",), 8),
        ("noise levels where more edges behind the relation help",
         ("noise_levels_where_more_edges_help",), 8),
        ("largest gain from accumulating edges", ("largest_gain",), 0.780),
        ("where the relation holds, at that largest gain",
         ("largest_gain_where_the_relation_holds",), 0.10),
        ("a relation holding 10%: right of all asked, from one edge",
         ("gains", 7, "from_one_edge"), 0.160),
        ("a relation holding 10%: right of all asked, at its best backing",
         ("gains", 7, "best"), 0.940),
        ("a relation holding 10%: the backing it is best at",
         ("gains", 7, "best_at_backing"), 23),
        ("a relation holding 100%: right of all asked, from one edge",
         ("gains", 0, "from_one_edge"), 0.880),
        ("a relation holding 100%: the backing it is best at",
         ("gains", 0, "best_at_backing"), 4),
        ("six edges, averaged over question seeds", ("six_edges_across_seeds",), 0.960),
        ("its spread over those seeds", ("six_edges_spread",), 0.014),
        ("every edge, averaged over the same seeds", ("every_edge_across_seeds",), 0.668),
        ("its spread over those seeds", ("every_edge_spread",), 0.045),
        ("how far apart the two are, in their own spreads", ("spreads_apart",), 6.244),
        ("rank of the relation from one edge", ("rank_at_one_edge",), 3.00),
        ("rank of the relation from twenty-three", ("rank_at_the_most_edges",), 0.625),
        ("a relation holding 10%, one edge: accuracy when it did answer",
         ("cells", 63, "accuracy"), 0.828),
        ("a relation holding 10%, one edge: share of questions answered",
         ("cells", 63, "answered_share"), 0.193),
        ("a relation holding 10%, twenty-three edges: accuracy when it answered",
         ("cells", 71, "accuracy"), 1.0),
        ("a relation holding 10%, twenty-three edges: share answered",
         ("cells", 71, "answered_share"), 0.940),
        ("the control, ignoring the relation, at a relation holding 10% and one edge",
         ("cells", 63, "without_the_operator"), 0.533),
    ],
    "exp35_weighting_read": [
        ("cosine weighting", ("cells", 0, "mrr_read"), 0.2029),
        ("association weighting", ("cells", 3, "mrr_read"), 0.1928),
        ("aperture weighting", ("cells", 4, "mrr_read"), 0.1868),
        ("the frequency floor", ("cells", 0, "mrr_unigram"), 0.1827),
        ("share of pairs cosine gives weight to", ("cells", 0, "live_share"), 0.672),
        ("share the association gives weight to", ("cells", 2, "live_share"), 0.976),
        ("share the aperture resolves", ("cells", 4, "live_share"), 0.216),
        ("the instrument's best over cosine", ("association_over_cosine",), -0.0102),
        ("that difference in standard errors",
         ("association_over_cosine_in_standard_errors",), -0.52),
        ("decisions", ("cells", 0, "tests"), 500),
        ("cache", ("cells", 0, "cache"), 500),
        ("cells clearing both tests", ("cells_clearing_both",), 0),
        ("cells swept", ("cells_swept",), 6),
    ],
    "exp31_rank_and_rarity": [
        ("unigram floor, by reciprocal rank", ("cells", 0, "mrr_unigram"), 0.1365),
        ("the lookup, by reciprocal rank", ("cells", 0, "mrr_n_gram"), 0.2376),
        ("one-hot operator", ("cells", 0, "mrr_operator"), 0.1536),
        ("counted operator, unit scaling", ("cells", 3, "mrr_operator"), 0.1792),
        ("counted operator, shared scaling", ("cells", 6, "mrr_operator"), 0.1650),
        ("standard error at this sample size", ("cells", 3, "se_operator"), 0.0049),
        ("rarest quartile: the operator", ("cells", 0, "rarest_operator"), 0.0114),
        ("rarest quartile: the lookup", ("cells", 0, "rarest_n_gram"), 0.0430),
        ("rarest quartile: the context-blind control",
         ("cells", 0, "rarest_rarity_control"), 0.0117),
        ("decisions", ("cells", 0, "tests"), 4000),
        ("cells where the operator outranks the lookup",
         ("cells_where_the_operator_outranks_the_lookup",), 0),
        ("cells where it outranks the lookup on the rarest quartile",
         ("cells_where_it_outranks_it_on_the_rarest_quartile",), 0),
        ("cells where it outranks the context-blind control on the rarest quartile",
         ("cells_where_it_also_outranks_the_rarity_control",), 0),
        ("cells swept", ("cells_swept",), 9),
        ("counted over one-hot", ("comparisons", "counted_over_one_hot"), 0.0256),
        ("counted over one-hot, in standard errors",
         ("comparisons", "counted_over_one_hot_in_standard_errors"), 3.78),
        ("operator over the frequency floor, in standard errors",
         ("comparisons", "operator_over_unigram_in_standard_errors"), 6.32),
        ("the lookup over the operator, in standard errors",
         ("comparisons", "lookup_over_operator_in_standard_errors"), 8.63),
    ],
    "exp19_wide_corpus": [
        ("same-activity, centred, in paired standard errors",
         ("results", 2, "edge_in_paired_standard_errors"), -2.32),
        ("same-activity, centred, separated",
         ("results", 2, "separated_at_two_standard_errors"), True),
        ("same-activity, raw, in paired standard errors",
         ("results", 0, "edge_in_paired_standard_errors"), -2.22),
        ("same-subject, centred, in paired standard errors",
         ("results", 3, "edge_in_paired_standard_errors"), -0.76),
        ("same-subject, centred, separated",
         ("results", 3, "separated_at_two_standard_errors"), False),
        ("ambient width", ("width",), 45),
        ("admission, same-activity relation", ("admission", 2, "margin_z"), 0.13),
        ("admission, same-subject relation", ("admission", 3, "margin_z"), -0.23),
        ("resolved modes per segment", ("facets", 1, "mean_modes"), 3.88),
        ("true candidate peak agreement", ("configuration", 1, "true_peak_agreement"), 0.726),
        ("distractor peak agreement", ("configuration", 1, "distractor_peak_agreement"), 0.553),
        ("questions with a louder distractor",
         ("configuration", 1, "questions_with_a_louder_distractor"), 0.667),
        ("generated: true peak agreement",
         ("generated_contrast", "true_peak_agreement"), 0.885),
        ("generated: distractor peak agreement",
         ("generated_contrast", "distractor_peak_agreement"), 0.910),
        ("generated: share of questions with a louder distractor",
         ("generated_contrast", "questions_with_a_louder_distractor"), 1.000),
        ("edge, same activity", ("edge_same_activity",), -0.063),
        ("edge, same subject", ("edge_same_subject",), -0.013),
    ],
    "exp20_audit_selection": [
        ("cells swept", ("cells",), 575),
        ("cells the generator cannot build", ("cells_the_generator_cannot_build",), 25),
        ("cells where z is perfect", ("cells_where_z_is_perfect",), 435),
        ("z on the published row", ("z_min_on_published_row",), 1.000),
        ("past the crossover, dot at or below chance",
         ("cells_past_crossover_where_dot_is_at_chance",), 260),
        ("cells where a scalar beats z", ("cells_where_a_scalar_beats_z",), 36),
    ],
    "exp23_audit_relation_count": [
        ("cells swept", ("cells",), 108),
        ("ceilings recovered exactly", ("ceilings_recovered_exactly",), 4),
        ("ceilings run", ("ceilings_run",), 4),
        ("largest kind count recovered exactly", ("largest_kind_count_recovered",), 64),
        ("the generator refuses past every ceiling",
         ("all_widths_refuse_past_the_ceiling",), True),
        ("cells the generator cannot build are counted, not dropped",
         ("quiet_cells_recovered_exactly",), 36),
    ],
    "exp25_crossover_strata": [
        ("questions", ("questions",), 900),
        ("median observed gain", ("median_gain",), 1.101),
        ("share with a louder distractor", ("share_with_a_louder_distractor",), 0.701),
        ("edge in the lowest-gain bin", ("edge_lowest_bin",), -0.533),
        ("edge in the highest-gain bin", ("edge_highest_bin",), -0.072),
        ("skill share in the highest-gain bin", ("skill_share_highest_bin",), 0.07),
        ("the skill share rises with the gain", ("skill_share_rises_with_gain",), False),
    ],
    "exp10_trained_baseline": [
        ("baseline fit on its own triples", ("baseline", "fit_on_training_triples"), 1.000),
        ("seen entities: trained", ("results", 0, "trained"), 0.905),
        ("seen entities: measured", ("results", 0, "measured"), 0.810),
        ("unseen entities: trained", ("results", 1, "trained"), 0.172),
        ("unseen entities: measured", ("results", 1, "measured"), 0.875),
    ],
}


def _dig(obj, path):
    """Follow a path of keys and indices; `"*"` collects over a list and returns the set."""
    for i, step in enumerate(path):
        if step == "*":
            rest = path[i + 1:]
            return {round(float(_dig(item, rest)), 6) for item in obj}
        obj = obj[step]
    return obj


def check_paths(names):
    """Resolve every registered path against the stored results, without re-running anything.

    A full run re-measures, which is the point of it — and takes long enough that a typo in a path
    is discovered an hour after it is made. This resolves each path against the JSON already on
    disk, so a mis-registered figure is caught in seconds. It says nothing about whether the numbers
    still reproduce; only that each check is asking the results a question they can answer.
    """
    results = HERE / "results"
    bad = stale = 0
    for name in names:
        short = name.split("_")[0]
        path = results / f"{short}.json"
        if not path.is_file():
            print(f"  {short:<10} no stored result; run it before checking paths")
            stale += 1
            continue
        out = json.loads(path.read_text(encoding="utf-8"))
        for label, keys, expected in CHECKS[name]:
            try:
                got = _dig(out, keys)
            except Exception as exc:                          # noqa: BLE001
                print(f"  {short:<10} UNRESOLVED  {label[:44]:<46} {exc}")
                bad += 1
                continue
            if isinstance(got, set):
                agrees = got == {round(float(expected), 6)}
            else:
                agrees = abs(float(got) - float(expected)) <= 5e-3
            if not agrees:
                shown = sorted(got) if isinstance(got, set) else f"{float(got):.4f}"
                print(f"  {short:<10} DISAGREES   {label[:44]:<46} "
                      f"published {expected} vs stored {shown}")
                bad += 1
    total = sum(len(CHECKS[n]) for n in names)
    print(f"\n{total} registered figures, {bad} that the stored results do not support, "
          f"{stale} experiments with no stored result")
    print("This checks the register against what is on disk. It does not re-measure: `verify.py`")
    print("with no `--paths` does that, and only that tells you the figures still reproduce.")
    return bad + stale


def verify(names):
    started = time.time()
    rows = []
    for name in names:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        mod = importlib.import_module(name)
        out = mod.run()
        if out is None:
            rows.append((name, "did not run", None, None, False))
            continue
        for label, path, expected in CHECKS[name]:
            try:
                got = _dig(out, path)
            except Exception as exc:                      # a path that no longer exists is drift
                rows.append((name, label, expected, f"path gone: {exc}", False))
                continue
            if isinstance(got, set):
                ok = got == {round(float(expected), 6)}
                shown = f"{sorted(got)}" if len(got) <= 3 else f"{len(got)} distinct values"
            else:
                ok = abs(float(got) - float(expected)) <= 5e-3
                shown = f"{float(got):.4f}"
            rows.append((name, label, expected, shown, ok))

    print(f"\n\n{'=' * 78}\nPUBLISHED FIGURES, RE-MEASURED\n{'=' * 78}")
    print(f"{'experiment':<30}{'figure':<42}{'published':>11}{'now':>14}  ")
    drift = 0
    for name, label, expected, got, ok in rows:
        mark = "" if ok else "   <-- DRIFT"
        exp = "" if expected is None else f"{expected}"
        print(f"{name.split('_')[0]:<30}{label[:41]:<42}{exp:>11}{str(got):>14}{mark}")
        drift += 0 if ok else 1
    print(f"\n{len(rows)} figures checked, {drift} drifted, in {time.time() - started:.0f}s")
    return drift


USAGE = """verify.py — re-measure every published figure against its own experiment.

  python research/verify.py                    re-measure everything (hours; use a configured host)
  python research/verify.py exp42 exp44        re-measure the experiments whose names contain these
  python research/verify.py --paths            check the register against results on disk (seconds)

A bare name is matched as a substring, so `exp42` selects `exp42_relation_vertex`.
"""

if __name__ == "__main__":
    args = sys.argv[1:]
    # An unknown flag is an error, and so is a name matching nothing. Dropping one silently turns a
    # typo into a full re-measure: `--help` would select nothing, fall back to everything, and start
    # hours of work on whatever machine it was typed on.
    known = {"--paths", "--help", "-h"}
    unknown = [a for a in args if a.startswith("-") and a not in known]
    if unknown:
        print(f"unrecognized: {' '.join(unknown)}\n\n{USAGE}")
        sys.exit(2)
    if {"--help", "-h"} & set(args):
        print(USAGE)
        sys.exit(0)

    paths_only = "--paths" in args
    wanted = [a for a in args if not a.startswith("-")]
    if wanted:
        chosen = [n for n in CHECKS if any(w in n for w in wanted)]
        missed = [w for w in wanted if not any(w in n for n in CHECKS)]
        if missed:
            print(f"no experiment matches: {' '.join(missed)}\n\n{USAGE}")
            sys.exit(2)
    else:
        chosen = list(CHECKS)
    runner = check_paths if paths_only else verify
    sys.exit(1 if runner(chosen) else 0)
