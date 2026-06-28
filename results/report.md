# JAX Interpretability Experiment Report

## Research Question

Can a small transformer trained from scratch in JAX/Flax be used as a controlled environment for mechanistic interpretability experiments?

More specifically:

- How do token predictions evolve across transformer layers?
- Which layers causally influence a target-token prediction?
- Can residual stream activations be decomposed into sparse feature representations?

---

## Experimental Setup

### Dataset

The model is trained on Tiny Shakespeare using character-level tokenization.

### Model

- 4-layer causal transformer
- 128-dimensional hidden states
- 4 attention heads
- 816,128 total parameters
- trained for 2,000 optimization steps

### Hardware

- NVIDIA T4 GPU
- Google Colab

### Frameworks

- JAX
- Flax
- Optax
- Orbax Checkpoint
- NumPy
- Matplotlib

---

## Training Results

Training and validation loss decreased steadily over 2,000 steps.

| Step | Train Loss | Validation Loss |
| ---: | ---------: | --------------: |
|  100 |     3.2079 |          3.2430 |
|  500 |     2.2808 |          2.3013 |
| 1000 |     2.0033 |          2.1033 |
| 1500 |     1.8307 |          1.9517 |
| 2000 |     1.8288 |          1.9237 |

The model did not show severe overfitting: training and validation loss tracked each other throughout training. This provides a stable foundation for interpretability analyses.

---

## Logit Lens Analysis

Logit Lens was used to project intermediate residual stream activations into vocabulary space.

The goal was to estimate what token the model would predict from each layer's representation.

### Observation

Prediction entropy jumps immediately after the embedding layer rather than changing gradually across all layers.

This is a more interesting mechanistic result than a smooth collapse would have been. It suggests that the embedding layer alone carries very little token-prediction information, and almost all of the decision-relevant transformation happens in the first transformer block.

In other words, the raw token embeddings are not yet strongly predictive on their own. The first transformer block appears to convert those embeddings into a much more prediction-relevant representation.

### Interpretation

This supports the hypothesis that early transformer layers perform a major representational transition from token identity to contextual prediction features.

This finding should not be overclaimed: the model is small and trained on character-level data. However, the sharp entropy change is useful because it gives a concrete mechanistic question for future work:

> What computation in the first transformer block causes the largest change in prediction confidence?

---

## Activation Patching Analysis

Activation patching was used to test causal influence across layers.

The experiment compared:

- a clean prompt
- a corrupted prompt
- patched runs where clean residual activations were inserted into the corrupted computation

### Observation

Later layers produced larger patch effects, with the final transformer layer showing the strongest causal contribution.

### Interpretation

This experiment used a 4-layer transformer, so "early/middle/late" should be read as a coarse 5-point scale (Layer 0 through Layer 4, where Layer 4 sits after the final transformer block) rather than a fine-grained progression. With that caveat, the pattern is still informative: causal effect on the target-token prediction grows monotonically with depth, and the largest effect appears at the final layer — the representation closest to the output is also the one doing the most direct work in recovering the corrupted prediction.

In a frontier-scale model with dozens or hundreds of layers, this same patching method is typically used to localize _which specific layers_ (often a narrow band, not "the last layer" in general) are responsible for a given behaviour — circuits research has found that meaningful computation is frequently concentrated in a handful of layers rather than spread evenly across depth. A 4-layer model can't show that kind of localization because there's no real "middle" to separate from "early" or "late" — every layer is adjacent to every other one. The monotonic increase observed here is consistent with depth mattering at all, but a model this shallow can't distinguish between "deeper layers matter more" and "there exists a specific small set of layers that matter most," which is the more interesting question at real scale. Testing that distinction is the natural next step — e.g. training a deeper model (8–16 layers) and checking whether the patch effect curve is still monotonic or whether it peaks and plateaus partway through, the pattern usually reported in larger models.

---

## Sparse Autoencoder Analysis

A Sparse Autoencoder was trained on residual stream activations from a middle transformer layer.

### Results

| Metric               |        Value |
| -------------------- | -----------: |
| Reconstruction MSE   |       0.0043 |
| Fraction zeros       |        0.330 |
| Mean active features | 685.7 / 1024 |

### Interpretation

The SAE successfully reconstructed residual activations with low mean squared error, but the learned representation was only moderately sparse.

The mean active feature count suggests that stronger sparsity regularization or a different SAE setup may be needed to produce more interpretable monosemantic features.

This is still useful as a first interpretability experiment because it demonstrates the full SAE workflow:

- collect residual activations
- train sparse feature representation
- measure reconstruction quality
- inspect feature frequency

---

## Key Findings

1. The transformer trained stably and reached a validation loss of 1.9237.

2. Logit Lens showed that prediction-relevant information appears sharply after the embedding layer, suggesting the first transformer block performs a major representational transformation.

3. Activation patching showed that later layers had larger causal effects on target-token recovery, though with only 4 layers this experiment can show that depth matters without being able to localize the effect to a specific sub-region of the network the way a deeper model would allow.

4. SAE analysis reconstructed residual activations well, but the feature representation was only moderately sparse.

5. The project provides a compact, reproducible environment for studying transformer internals.

---

## Limitations

- Character-level dataset rather than modern tokenized language modeling.
- Small model size limits generalization to frontier-scale transformers.
- Single training run rather than multi-seed analysis.
- SAE sparsity was moderate, not strongly selective.
- Activation patching was tested on a small number of prompts.
- With only 4 layers, activation patching can establish that depth matters but cannot test whether causal effect is concentrated in a specific sub-region of the network, which is the more common finding in deeper models.

---

## Future Work

- Run activation patching across many prompts.
- Add attention-head patching.
- Increase SAE sparsity and compare feature quality.
- Add induction-head style tasks.
- Compare layerwise entropy patterns across model sizes.
- Build feature dashboards for top SAE activations.
- Train a deeper model (8–16 layers) and test whether the patch effect curve is still monotonic or whether it peaks and plateaus partway through, as is commonly reported at larger scale.

---

## Conclusion

This project shows that a small transformer trained from scratch can support meaningful mechanistic interpretability experiments. The most interesting result is that prediction entropy changes sharply after the embedding layer, suggesting that the first transformer block performs a major decision-relevant transformation. Activation patching showed a monotonic increase in causal effect with depth, consistent with depth mattering, though a 4-layer model cannot test the more specific and more commonly reported finding in larger models — that causal effect is often concentrated in a narrow band of layers rather than increasing smoothly throughout. Combined with sparse autoencoder analysis, the project provides a strong foundation for deeper transformer interpretability work, with a clear next step toward a deeper model to test that distinction directly.
