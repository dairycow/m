# m — SWE-bench Lite results

Model: **Gemma 4 12B (Q5_K_XL) + MTP drafter** via llama.cpp on an RTX 4070 Ti SUPER, agent: **m** (headless `-p` mode, temp 0, max 40 turns).

| metric | value |
|---|---|
| **resolved** | **11/30** (36.7%) |
| patch generated | 20/30 |
| total wall time | 1h01m |
| mean turns | 21.9 |
| mean generation speed | 133 tok/s |

| instance | outcome | turns | time | patch |
|---|---|---|---|---|
| astropy__astropy-12907 | — no patch | 13 | 2m50s | 0 B |
| django__django-11039 | ✅ resolved | 22 | 1m40s | 657 B |
| django__django-11630 | ❌ not resolved | 22 | 2m26s | 2966 B |
| django__django-12125 | — no patch | 40 | 1m11s | 0 B |
| django__django-12708 | — no patch | 40 | 3m44s | 0 B |
| django__django-13230 | ✅ resolved | 40 | 2m15s | 594 B |
| django__django-13660 | ✅ resolved | 12 | 1m10s | 550 B |
| django__django-14238 | ✅ resolved | 24 | 1m59s | 722 B |
| django__django-14787 | ❌ not resolved | 9 | 2m14s | 706 B |
| django__django-15347 | ✅ resolved | 12 | 0m55s | 710 B |
| django__django-15819 | — no patch | 31 | 3m09s | 0 B |
| django__django-16400 | — no patch | 28 | 3m31s | 0 B |
| matplotlib__matplotlib-18869 | ❌ not resolved | 10 | 0m32s | 462 B |
| matplotlib__matplotlib-23987 | ✅ resolved | 31 | 1m02s | 680 B |
| matplotlib__matplotlib-25498 | — no patch | 12 | 3m31s | 0 B |
| psf__requests-1963 | ✅ resolved | 14 | 2m05s | 360 B |
| pydata__xarray-5131 | ✅ resolved | 8 | 0m41s | 570 B |
| pytest-dev__pytest-5221 | — no patch | 14 | 2m14s | 0 B |
| pytest-dev__pytest-7490 | ❌ not resolved | 18 | 1m45s | 648 B |
| scikit-learn__scikit-learn-13142 | ✅ resolved | 13 | 1m18s | 1066 B |
| scikit-learn__scikit-learn-14983 | ❌ not resolved | 25 | 2m29s | 505 B |
| sphinx-doc__sphinx-7686 | ❌ not resolved | 25 | 2m37s | 2284 B |
| sphinx-doc__sphinx-8713 | ❌ not resolved | 40 | 0m41s | 1450 B |
| sympy__sympy-12481 | ✅ resolved | 16 | 2m02s | 884 B |
| sympy__sympy-13895 | — no patch | 11 | 2m42s | 0 B |
| sympy__sympy-15308 | ❌ not resolved | 26 | 1m53s | 547 B |
| sympy__sympy-17022 | — no patch | 40 | 1m44s | 0 B |
| sympy__sympy-18698 | — no patch | 14 | 3m19s | 0 B |
| sympy__sympy-20590 | ✅ resolved | 32 | 1m26s | 311 B |
| sympy__sympy-22714 | ❌ not resolved | 14 | 2m17s | 692 B |

Scoring: official harness — `python -m swebench.harness.run_evaluation --dataset_name SWE-bench/SWE-bench_Lite --predictions_path bench/runs/v1/predictions.jsonl --run_id m-bench --max_workers 4`
