# Mixed flocks in open space

Code for the paper *Segregation with cohesion: a quantum-inspired perceptual model of mixed flocks in open space*, by Mayank Chaurasiya, Ayush Agrawal, Chirayush Mohanty and Laxmidhar Behera (Indian Institute of Technology Mandi and Indian Institute of Technology Kanpur).

Agents of three kinds move at constant speed in an open plane. Each agent perceives six partners, chosen in proportion to how similar they are to it, and holds their headings and bearings in one perceptual state. The bearing of a partner of another kind enters with a minus sign. The agent draws its next heading from the squared perceptual state (Born rule). With these two ingredients the kinds sort into single-kind subgroups while the flock stays together.

## What is here

| File | What it does |
|---|---|
| `paper.ipynb` | Every simulation, table and figure of the paper, in order. Start here. |
| `flock_model.py` | The model as a Python module (the same code as Sections 1–4 of the notebook), used by the scripts in `checks/`. |
| `checks/lopsided_draw.py` | Section 3.3: how much the minus sign shifts the draw away from the other kind. |
| `checks/subgroup_gaps.py` | Section 5.5: whether the subgroups drift apart over time (signed vector sum, vector sum with other kinds' bearings ignored, full model). |
| `checks/measures_k.py` | Appendix B: the mixing entropy for 5, 10, 20 and 40 neighbours, and cohesion for other link distances. |
| `checks/appendix_math.py` | Appendices C and D: the closed forms of the overlaps against numerical integrals, and the worked example. |

## Install

Python 3.9 or newer.

```
pip install -r requirements.txt
```

`ffmpeg` is needed only for the video in the last section of the notebook.

## Run

**Check first.** In `paper.ipynb`, Section 1, set `QUICK_TEST = True` and run all cells. This uses tiny flocks and short runs and checks that every cell works in a few minutes. The numbers it prints are not the paper's.

**Full run.** Set `QUICK_TEST = False`, set `N_JOBS` to the number of CPU cores to use, and run all cells. All tables, figures and data go to `output/`.

One run of 5,000 steps takes about 25 s for 240 agents, 80 s for 480 and 4 min for 960, on one core. The whole notebook is about 20 core-hours, so about 2 hours on 10 cores. Every run is saved as soon as it finishes, under a key made from all of its parameters, so the notebook can be stopped and restarted and only what is missing is computed. Section 7 (the large flocks) is half of the time; to shorten it, remove the entry `960` from `LIMIT_PLAN`.

**Checks.** From the repository folder:

```
python checks/appendix_math.py        # a few seconds
python checks/lopsided_draw.py 60     # a few minutes
python checks/subgroup_gaps.py        # about 10 minutes on 4 cores
python checks/measures_k.py           # about 10 minutes on 4 cores
```

Set the number of cores with `N_JOBS`, for example `N_JOBS=8 python checks/subgroup_gaps.py`.

## Where each result comes from

| Paper | Notebook section or script |
|---|---|
| Figure 1, Appendix D | Section 14 |
| Table 2, Figures 2 and 3 | Section 5 |
| Table 3 (decision rules) and its tests | Section 6 |
| Table 4 (cues), the flock of 960 agents | Section 11 |
| Figure 4, Table A6 (range of sight) | Section 7 |
| Figure 5 (attention to other kinds) | Section 8 |
| Figure 6 (noise) | Section 9 |
| Figure 7 (where the result holds) | Section 10 |
| Figure 8, Appendix E (perception strength) | Section 12 |
| Figure 9 (long runs) | Section 13 |
| Vicsek-like limit (Section 3.4) | Section 15 |
| Every number quoted in the text | Section 16, `output/numbers_for_manuscript.txt` |
| Lopsided draw (Section 3.3) | `checks/lopsided_draw.py` |
| Drift of the subgroups (Section 5.5) | `checks/subgroup_gaps.py` |
| Choice of the measures (Appendix B) | `checks/measures_k.py` |
| Closed forms and worked example (Appendices C, D) | `checks/appendix_math.py` |

## Seeds

Tables 2 and 3 use seeds 3000–3029 (the same 30 starting states for every case). The other sweeps use seeds 2000–2019, and the range-of-sight runs start at 4000. All seeds are set in Section 1 of the notebook.

## Citation

If you use this code, please cite the paper (reference to be added on publication).

## Licence

MIT, see `LICENSE`.
