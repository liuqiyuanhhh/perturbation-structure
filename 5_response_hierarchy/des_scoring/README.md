# DES scoring (placeholder)

Missing: the per-perturbation DES scorer, its residual-effect threshold sweep
(`des_threshold.py`) and the builders of the truth tables they score against.
They serve Fig 2b, 2d, S3c and S5b; `../DES_figures.ipynb` summarizes their two tables.

What the step does, for each screen, CV fold (seed 1-5), method and held-out
perturbation, on the saved predictions of `4_prediction_benchmark`
(`data/prediction_result/{screen}_predictions_by_seed.pkl`, `_splits_by_seed.pkl`):

- DES is the overlap between the k true DE genes and the method's top k genes
  by absolute predicted effect, divided by k; the direct target is removed
  from both. `rejfreq` ranks genes by their DE frequency in the training
  perturbations (ties in column order), `random` draws k genes without
  replacement (`numpy.random.default_rng(seed)`, `seed=` the fold).
- `before_detrend`: the raw predictions against the total-effect truth.
  `after_detrend`: each prediction block loses its own leading factor
  (`remove_first_factor(frame, random_state=0, zero_tol=1e-5)`) and is scored
  against the residual-effect (max-p detrended) truth, whose leading factor
  is fitted with the direct-target entries zeroed; `train_mean` is not scored
  here.
- Only perturbations present in at least one source screen of the target are
  scored (the source pool of `4_prediction_benchmark/targets.py`).
- The threshold sweep repeats `after_detrend`, restricting the truth to the
  discoveries with detrended |effect| above each cutoff (0, 0.01, 0.05,
  0.10, ..., 0.50).

It must write, in `data/external/des/`:

- `des_per_pert.csv`: `dataset, seed, space, method, perturbation, k, des`,
  for `rejfreq, weighted, presage, linear_train, linear_otherds, gears,
  scGPT-ft, train_mean, random` (`linear_train` and `linear_otherds` are the
  prediction keys `paper_linear_embedding_from_training_10` and
  `linear_embedding_from_otherds_10`); `k` is the number of true discoveries,
  `des` is empty when `k = 0`;
- `des_threshold_per_pert.csv`: `dataset, threshold, seed, method,
  perturbation, k, des`, for the same methods except `train_mean`.
