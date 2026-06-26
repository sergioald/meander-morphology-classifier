# Zenodo release checklist

Use this checklist before creating the Zenodo archive of the GitHub/software repository.

## A. Before release

- [ ] `pytest` passes.
- [ ] `python -m compileall src scripts app` passes.
- [ ] README includes the single-bend and compound-bend workflows.
- [ ] `CITATION.cff` is present and valid.
- [ ] `.zenodo.json` is present and contains software metadata.
- [ ] `LICENSE` is present.
- [ ] Model files are not tracked by Git.
- [ ] Generated `outputs/` files are not tracked by Git.
- [ ] Documentation explains where to download the trained compound model/data files.
- [ ] The manuscript Data and Software Availability Statement cites both Zenodo records.

## B. Recommended Git release

Use a semantic release tag, for example:

```bash
git tag -a v0.2.0 -m "Compound bend spectral segmentation and latent encoding release"
git push origin v0.2.0
```

Then create a GitHub Release from the tag. If GitHub is linked to Zenodo, Zenodo can archive the GitHub release.

## C. Software Zenodo record

Recommended metadata:

```text
Title: Meander Morphology Classifier
Resource type: Software
Creators: Sergio Lopez Dubon; Alessandro Sgarabotto; Stefano Lanzoni
License: MIT
Keywords: meandering rivers; compound meanders; curvature; continuous wavelet transform; autoencoder; geomorphology; machine learning
```

Suggested description:

```text
Python research-software toolkit and Streamlit graphical user interface for curvature-spectral meander bend classification. The software supports inflection-point single-bend extraction, CWT spectrum generation, compound bend segmentation from curvature spectral energy, and latent-space inference using separately archived trained model files.
```

## D. Model/data Zenodo record

Recommended metadata:

```text
Title: Trained autoencoder and encoder for curvature-spectral analysis of compound meander bends
Resource type: Software, or Dataset if the record is treated mainly as model/data artefacts
Creators: Sergio Lopez Dubon; Alessandro Sgarabotto; Stefano Lanzoni
```

Main files:

```text
trained_autoencoder.h5
encoder_only.h5
encoder_only.keras
model_architecture.json
encoder_architecture.json
model_summary.txt
encoder_summary.txt
world_latent_cloud.npy
world_latent_cloud_metadata.json
model_card.md
```

## E. Related identifiers

After both records have DOIs, add related identifiers in Zenodo:

Software record:

```text
Is documented by: paper DOI or preprint DOI
Requires / Is supplemented by: model/data Zenodo DOI
```

Model/data record:

```text
Is documented by: paper DOI or preprint DOI
Is supplemented by / Is required by: software Zenodo DOI
```

## F. Manuscript text

Replace the manuscript Data Availability Statement with a Data and Software Availability Statement that cites:

1. the software/GUI Zenodo DOI;
2. the model/data Zenodo DOI;
3. the paper DOI or preprint DOI, once available.
