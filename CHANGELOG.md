# Changelog

All notable model versions are documented here.

## [Unreleased]

## [1.0.0] - 2025-03-09

### Added

- **First release**: Ultra-Sortformer 8spk
- Source: `best_09884.ckpt` (val_f1_acc=0.9884)
- Training: 7000 sessions, 180s, 2–8 speakers
- Base: NVIDIA Sortformer 4spk extended to 8spk

---

## Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.x.x): Architecture change, breaking compatibility
- **MINOR** (x.1.x): Improved performance, new training run
- **PATCH** (x.x.1): Bug fix, minor tweak

### How to update

1. Train new model → get best checkpoint
2. Convert to `.nemo` if needed
3. Upload to Hugging Face (overwrites or add as new file)
4. Update this CHANGELOG
5. Update README "Model Versions" table
6. (Optional) Create git tag: `git tag v1.1.0`
