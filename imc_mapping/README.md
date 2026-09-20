# imc_mapping/

**Contributors:** Raya Chauhan.

Maps the `OptionPriceNN` surrogate (5 → 64 → 64 → 1) onto a conventional
von Neumann architecture vs. an in-memory-computing (IMC) architecture,
as a software-level proof of concept (no physical chip required).

## What's here

- `imc_model.py` — computes MAC counts, memory accesses, and data
  movement per dense layer (`y = Wx + b`); converts those into energy
  and latency estimates for both architectures; appends result rows to
  `results/results.csv` using the shared schema.
- `ASSUMPTIONS.md` — every estimate used (energy per MAC, energy per
  byte moved, bandwidths), where each number comes from, and the
  headline results with the key caveat about single-sample vs. cached
  weights.

## Run it

```
cd imc_mapping
python imc_model.py
```

Optionally point at a trained model once one exists:

```
python imc_model.py --model ../neural_network/best_model.npz
```

## Headline result

Single-sample inference: IMC cuts data movement by ~94.6% and energy by
~99.8% vs. a conventional architecture that re-fetches all weights from
DRAM every inference — the realistic worst case for edge/streaming
deployment, and the scenario IMC is built to help with. See
`ASSUMPTIONS.md` §4 for the numbers and the caveat about weight caching.

## Handoff → benchmarking/

`results/results.csv` now has:
- `neural_network_conventional_datamovement` (fills the data-movement /
  energy fields `neural_network/evaluate.py` leaves blank)
- `neural_network_imc` (the full IMC-side estimate)

Ishika can pull both straight into the final MC vs. NN-CPU vs. NN+IMC
comparison table/plots.
