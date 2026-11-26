# CombinatoryPlaySol

A frozen conceptual matrix with a continuous recurrent reader. The next **workspace**, not the next token, is the unit of operation. This is an exploratory instrument; no claim of consciousness, fractality, or improved reasoning.

Companion work: [SweepSol](https://github.com/anttiluode/SweepSol), [Sweep](https://github.com/anttiluode/Sweep), [Unlock experiment](https://github.com/anttiluode/Sweep/tree/main/runs/unlock_qwen), [developmental experiment](https://github.com/anttiluode/Sweep/tree/main/runs/develop_pythia-410m_h20). The independent Claude version is not modified here.

## Run now

```bat
git clone https://github.com/anttiluode/CombinatoryPlaySol
cd CombinatoryPlaySol
start index.html
```

Open the existing SweepSol/bank.json from the file picker. No installation, server, GPU, or new model pass. The toy loaded on startup is explicitly synthetic. The real bank contains PCA-compressed sampled word states, **not J-lens vectors** or the full model. Start with bird,shadow,winter or your own cues. Step or Run. A gold point is the continuous state; blue/green points are simultaneously active observations. Green means outside all initial single-cue full-space top-k neighborhoods. The 2D chart is only a projection. The model's actual weights are not loaded.

The state, query, workspace and memory gate are logged separately. Change grain, feedback, cue tether, coverage and temperature; Reset applies controls. Apply tag changes the slow context without resetting the current state. Example episodes are fictional and hand-authored; they test a mechanism, not learned autobiographical memory. They can be edited locally. Do not publish private episode data in traces.

## The operation

Let x be a continuous normalized vector in the cached space, c the normalized sum of initial cues, and t a separate persistent context vector. Optional episode records have key, tag and value vectors; a record is active when both cosine(x,key) and cosine(t,tag) exceed the threshold (0.2 by default). Their normalized weighted value sum is m. The query is normalize(x + tagBias*t + memoryStrength*m). Direct tag bias defaults to zero.

Read the top k candidate word states at the selected grain, subtracting a decaying coverage penalty. Softmax weights on this selected workspace are proportional to exp(score/T). Let y be the normalized weighted sum of those observations in the full cached space. Update:

    x_next = normalize((1-feedback-anchor)*x + feedback*y + anchor*c)

The state is never snapped to a word in recurrent mode. The feedback is through the whole weighted workspace, not a generated word. No target, hidden reward, online training, or token emission. Initial cue words are excluded from workspace selection in every arm. Temperature rises when consecutive workspace overlap exceeds 0.6 and cools otherwise; it changes mixing weights, **not top-k selection directly**, unlike sampling/acceptance in Sweep. Grain can be fixed or cycled (not an adaptive policy). A continuous state is not guaranteed to be on the distribution of natural model states.

## Compare matched modes

Node.js, no dependencies:

```bat
node run.js --self-test
node run.js --bank demo --ticks 20 --k 8
node run.js --bank ..\SweepSol\bank.json --cues bird,shadow,winter --ticks 40 --out play-qwen.json
node run.js --bank ..\SweepSol\bank.json --cues bird,shadow,winter --tag music --episodes example-episodes.json --switch-tag family --switch-at 20 --ticks 40 --out play-context.json
```

Five arms use identical initial cues, workspace size and beat count: static blend (no coverage, tag or feedback), no-feedback (fixed cue state, same coverage/context machinery), discrete top-word commit, continuous recurrent, continuous recurrent without context. Use the no-feedback arm to isolate the value of recurrent state over coverage and context alone. The extra context-switch arm is optional. State intervention: derive two states from recurrent and no-feedback histories, inject each into otherwise identical new receivers with the same prior workspace, tag cleared, coverage zero, and report next-workspace Jaccard. This is a causal wiring test, not an intelligence test.

Reports include distinct observations, distinct outside-single-cue neighbors, mean full-space coherence, cue retention, overlap, state motion and episode activation. A high coherence score can mean a trap, and high novelty can mean noise. They are descriptive, not optimization targets. Recursive unlocking is a hypothesis; this implementation does not prove a hierarchy or fractal scaling.

Optional native text comparison (requires torch/transformers and model available locally, use --local-files-only to prevent downloads):

```bat
python3.13 compare_qwen.py --play play-qwen.json --device cuda --local-files-only
```

Generates literal continuations and prompted associations using the same cue words; an optional tagged prompt is included if a tag exists. Saves full text and lexical overlap against each play arm. Different outputs alone are not evidence of superior cognition. This is not a matched computational or information budget; the model has its own full weights and is not given episode values. A future useful task needs independent held-out judgments of relevance and usefulness.

## Receipts

Seven synthetic wiring checks passed: weighted parallel workspace, unsnapped state, static stability, frozen bank, context-selective episode activation, state transfer and matched arms. On the deliberately hand-designed toy, 20 beats with k=8 gave 8 distinct observations (static), 13 (no-feedback), 13 (discrete) and 15 (recurrent). Same-prior-workspace state intervention gave state cosine 0.9224 and next-workspace Jaccard 0.7778. These are toy outcomes, not real Qwen results. Browser inline and CLI syntax parsed; browser end-to-end, Python comparison and real-bank behavior were not executed here.

The main question is whether recurrent feedback over multiple concurrent associations contributes useful structure beyond static blending and coverage, not whether an external reader can produce different words from a language model.


## Recoverable energy / fatigue (real-bank trap report)

The first user Qwen trace (bird,shadow,winter, 516 beats, k=12, full cached grain) visited 79 distinct words; the last new word occurred at beat 91. For beats 280–516 only 25 distinct addresses were visited, with mean consecutive workspace Jaccard 0.889 and mean state motion 1-cosine 0.00168. The last 12 words were cow, bat, bull, crow, fly, horse, bear, dog, cat, rat, mouse, bug. Temperature was at its cap (0.36). Heating only changes weights **after** top-k selection, so it cannot directly admit new words. Original coverage adds 1/k per selected word and decays by 0.10; its maximum steady penalty at the default settings is 0.16/(12*0.10) = 0.1333.

New optional fatigue acts **before** selection. Each bank address starts with energy E=1. Selection score is old score minus fatigueStrength*(1-E). After reading, active addresses consume energy, and unselected addresses can also consume energy proportionally to positive full-space cosine to the weighted workspace centroid above the spread threshold. Energy recovers by recovery*(1-E) per beat, while burning burn*demand; E is clipped to [0,1]. Spread=1 means strict address-only fatigue. This is a computational adaptation analogy, not a model of neuronal ATP depletion.

Browser: set Fatigue strength to 1.00, Burn 0.12, Recovery 0.015, Spread 0.55 and Reset. The default fatigue strength is 0 so previous behavior is preserved. Watch active energy and distinct concepts. Compare the same run with spread=1 or fatigue=0. The CLI has a matched three-arm comparison:

```bat
node run.js --bank ..\SweepSol\bank.json --cues bird,shadow,winter --ticks 516 --fatigue-test 1 --out play-fatigue.json
```

All arms retain the same coverage, temperature, cue tether and feedback. The arms are baseline, address-only, and regional fatigue. Report last novel beat, distinct words, coherence, retention, overlap and motion together. Fatigue is not a guarantee of continual meaningful novelty; finite banks saturate and moving further can lose relevance.

Final committed JavaScript synthetic checks passed, including depletion, regional spread, recovery and disabled regression. With the toy bank (300 beats, k=8), baseline observed 15 distinct words (mean coherence 0.836), address-only 45 (0.705), regional 45 (0.716). Both fatigue arms visited all 45 eligible words, so novelty saturates on this toy, and regional spread did not improve distinct count over address-only. Real Qwen fatigue performance and browser end-to-end behavior remain untested here. No user trace or private memories were committed.
