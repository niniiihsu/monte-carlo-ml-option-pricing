# monte_carlo/

**Contributors:** Nini Hsu (original pricer + dataset), Ishika Pandurangam (module refactor, docs, validation writeup). Open to anyone on the team — not exclusive.

Covers:
- European call Monte Carlo pricer
- Black-Scholes validation (S=100, K=100, T=1, sigma=0.20, r=0.05 -> ~$10.4506; MC result ~$10.4634, ~0.12% error)
- Synthetic scenario generation (moneyness S/K in 0.7-1.3)
- Dataset CSV output (see root README for schema)
- Runtime/num_paths logging

Handoff -> neural_network/: final dataset CSV in agreed format.
