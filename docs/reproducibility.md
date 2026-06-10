# Reproducibility

The repository separates three levels of reproducibility:

1. **Lightweight tests** using synthetic arrays and a small synthetic centerline.
2. **Workflow reproduction** using user-provided or public centerline data and the Zenodo autoencoder.
3. **Full research reproduction** using the complete original datasets and model-training workflow.

The first level runs in CI. The second and third levels are intentionally kept outside CI because they require large data and/or TensorFlow.
