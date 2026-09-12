# Landsat recovery model

This phase models event-level Landsat NBR recovery using
`outputs/reference_timeseries/disturbed_reference_event_year.csv`.

The primary response is:

`nbr_gap_change = (NBR_reference - NBR_disturbed) - pre_nbr_gap`

where `pre_nbr_gap` is the event-level median NBR gap from available years -3,
-2, and -1. The final primary model is a spline-based nonlinear trajectory
model with group/event cross-validation and event-bootstrap prediction
intervals. The linear model is retained as a simple comparator.

No new Landsat, Sentinel-2, or AlphaEarth data were extracted in this phase.

```json
{
  "input": "outputs/reference_timeseries/disturbed_reference_event_year.csv",
  "complete_exact_year_fire_harvest_pairs": 76,
  "events_with_pre_baseline": {
    "fire_total": 72,
    "harvest_total": 69
  },
  "post_model_events": {
    "fire_total": 68,
    "harvest_total": 69
  },
  "post_model_event_years": 1489,
  "functional_form_selected_primary": "nonlinear_spline",
  "linear_cv_rmse": 0.18997986206166748,
  "nonlinear_cv_rmse": 0.18122343200910743,
  "nonlinear_selected_alpha": 0.01,
  "nonlinear_effective_degrees_of_freedom": 14.305581034670201,
  "fire_harvest_ci_first_overlaps_zero_age": 1,
  "nonlinear_fixed_age_fire_minus_harvest": [
    {
      "years_since_disturbance": 0,
      "fire_estimate": 0.497113450921209,
      "harvest_estimate": 0.21576680174546872,
      "fire_minus_harvest": 0.28134664917574026,
      "ci95_low_conservative": 0.11319250907999323,
      "ci95_high_conservative": 0.43515730938997765,
      "ci_overlaps_zero": false
    },
    {
      "years_since_disturbance": 1,
      "fire_estimate": 0.40066698622550107,
      "harvest_estimate": 0.32975510449565026,
      "fire_minus_harvest": 0.0709118817298508,
      "ci95_low_conservative": -0.04064537840463306,
      "ci95_high_conservative": 0.19516268872855602,
      "ci_overlaps_zero": true
    },
    {
      "years_since_disturbance": 5,
      "fire_estimate": 0.20646873974626676,
      "harvest_estimate": 0.13587459701213744,
      "fire_minus_harvest": 0.07059414273412931,
      "ci95_low_conservative": -0.016634357568657637,
      "ci95_high_conservative": 0.1555386519109578,
      "ci_overlaps_zero": true
    },
    {
      "years_since_disturbance": 10,
      "fire_estimate": 0.10124413801621486,
      "harvest_estimate": 0.027096689747726402,
      "fire_minus_harvest": 0.07414744826848846,
      "ci95_low_conservative": 0.011920854458558117,
      "ci95_high_conservative": 0.14264510410733286,
      "ci_overlaps_zero": false
    },
    {
      "years_since_disturbance": 15,
      "fire_estimate": 0.016334426879412507,
      "harvest_estimate": 0.006496967181544675,
      "fire_minus_harvest": 0.009837459697867833,
      "ci95_low_conservative": -0.08147053562652376,
      "ci95_high_conservative": 0.09900930430810599,
      "ci_overlaps_zero": true
    },
    {
      "years_since_disturbance": 20,
      "fire_estimate": -0.037203629233981815,
      "harvest_estimate": 0.02431275094322957,
      "fire_minus_harvest": -0.06151638017721138,
      "ci95_low_conservative": -0.2670244451639482,
      "ci95_high_conservative": 0.12144958644656742,
      "ci_overlaps_zero": true
    }
  ],
  "raw_nbr_gap_fixed_age_predictions": [
    {
      "model": "nbr_gap_median",
      "disturbance_type": "fire_total",
      "years_since_disturbance": 0,
      "estimate": 0.5381680692898464,
      "ci95_low": 0.4546374444946411,
      "ci95_high": 0.6255449623048693,
      "bootstrap_iterations": 600,
      "alpha": 0.01,
      "effective_degrees_of_freedom": 14.305581034670201
    },
    {
      "model": "nbr_gap_median",
      "disturbance_type": "fire_total",
      "years_since_disturbance": 5,
      "estimate": 0.2548302181017508,
      "ci95_low": 0.22117442601922951,
      "ci95_high": 0.29534936724296706,
      "bootstrap_iterations": 600,
      "alpha": 0.01,
      "effective_degrees_of_freedom": 14.305581034670201
    },
    {
      "model": "nbr_gap_median",
      "disturbance_type": "fire_total",
      "years_since_disturbance": 10,
      "estimate": 0.1435704776381581,
      "ci95_low": 0.11075961558248666,
      "ci95_high": 0.18070693028539445,
      "bootstrap_iterations": 600,
      "alpha": 0.01,
      "effective_degrees_of_freedom": 14.305581034670201
    },
    {
      "model": "nbr_gap_median",
      "disturbance_type": "fire_total",
      "years_since_disturbance": 15,
      "estimate": 0.06185278391581687,
      "ci95_low": 0.027794069317009475,
      "ci95_high": 0.09636926759452108,
      "bootstrap_iterations": 600,
      "alpha": 0.01,
      "effective_degrees_of_freedom": 14.305581034670201
    },
    {
      "model": "nbr_gap_median",
      "disturbance_type": "fire_total",
      "years_since_disturbance": 20,
      "estimate": 0.023495864000105473,
      "ci95_low": -0.046128618715984164,
      "ci95_high": 0.1233685441683619,
      "bootstrap_iterations": 600,
      "alpha": 0.01,
      "effective_degrees_of_freedom": 14.305581034670201
    },
    {
      "model": "nbr_gap_median",
      "disturbance_type": "harvest_total",
      "years_since_disturbance": 0,
      "estimate": 0.23084707674790736,
      "ci95_low": 0.17039446977467848,
      "ci95_high": 0.32103344310808035,
      "bootstrap_iterations": 600,
      "alpha": 0.01,
      "effective_degrees_of_freedom": 14.305581034670201
    },
    {
      "model": "nbr_gap_median",
      "disturbance_type": "harvest_total",
      "years_since_disturbance": 5,
      "estimate": 0.1499501156022563,
      "ci95_low": 0.10679042345130449,
      "ci95_high": 0.20056848095463475,
      "bootstrap_iterations": 600,
      "alpha": 0.01,
      "effective_degrees_of_freedom": 14.305581034670201
    },
    {
      "model": "nbr_gap_median",
      "disturbance_type": "harvest_total",
      "years_since_disturbance": 10,
      "estimate": 0.04325698968976628,
      "ci95_low": 0.014547320026235563,
      "ci95_high": 0.07210043454932347,
      "bootstrap_iterations": 600,
      "alpha": 0.01,
      "effective_degrees_of_freedom": 14.305581034670201
    },
    {
      "model": "nbr_gap_median",
      "disturbance_type": "harvest_total",
      "years_since_disturbance": 15,
      "estimate": 0.029006918046093427,
      "ci95_low": -0.01942958649111639,
      "ci95_high": 0.08367574602877462,
      "bootstrap_iterations": 600,
      "alpha": 0.01,
      "effective_degrees_of_freedom": 14.305581034670201
    },
    {
      "model": "nbr_gap_median",
      "disturbance_type": "harvest_total",
      "years_since_disturbance": 20,
      "estimate": 0.0543233758197439,
      "ci95_low": -0.020693201425028833,
      "ci95_high": 0.1706450311528905,
      "bootstrap_iterations": 600,
      "alpha": 0.01,
      "effective_degrees_of_freedom": 14.305581034670201
    }
  ],
  "paired_model_pairs": 67,
  "paired_support_fixed_ages": [
    {
      "years_since_disturbance": 0,
      "n_fire_events": 49,
      "n_harvest_events": 52,
      "n_complete_fire_harvest_pairs": 41,
      "n_fire_event_years": 49,
      "n_harvest_event_years": 52
    },
    {
      "years_since_disturbance": 1,
      "n_fire_events": 51,
      "n_harvest_events": 58,
      "n_complete_fire_harvest_pairs": 46,
      "n_fire_event_years": 51,
      "n_harvest_event_years": 58
    },
    {
      "years_since_disturbance": 5,
      "n_fire_events": 41,
      "n_harvest_events": 38,
      "n_complete_fire_harvest_pairs": 31,
      "n_fire_event_years": 41,
      "n_harvest_event_years": 38
    },
    {
      "years_since_disturbance": 10,
      "n_fire_events": 33,
      "n_harvest_events": 36,
      "n_complete_fire_harvest_pairs": 28,
      "n_fire_event_years": 33,
      "n_harvest_event_years": 36
    },
    {
      "years_since_disturbance": 15,
      "n_fire_events": 33,
      "n_harvest_events": 31,
      "n_complete_fire_harvest_pairs": 29,
      "n_fire_event_years": 33,
      "n_harvest_event_years": 31
    },
    {
      "years_since_disturbance": 20,
      "n_fire_events": 9,
      "n_harvest_events": 12,
      "n_complete_fire_harvest_pairs": 9,
      "n_fire_event_years": 9,
      "n_harvest_event_years": 12
    }
  ],
  "negative_late_age_observation_support": [
    {
      "disturbance_type": "fire_total",
      "event_years": 44,
      "median": -0.05880225,
      "p25": -0.142524375,
      "p75": 0.036212875000000005,
      "proportion_negative": 0.7272727272727273
    },
    {
      "disturbance_type": "harvest_total",
      "event_years": 51,
      "median": -0.013608000000000002,
      "p25": -0.1008665,
      "p75": 0.08655550000000001,
      "proportion_negative": 0.6078431372549019
    }
  ],
  "method_note": "Spline trajectories use event-level rows, event-grouped cross-validation, and event-bootstrap prediction intervals; linear model remains the simple comparator."
}
```
