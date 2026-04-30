---
title: MedVision Demo
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
---

# MedVision Demo

Cell Segmentation & Foci Classification for microscopy images.

## Features

- **Cell Segmentation** — Cellpose-based segmentation with configurable model, diameter, flow threshold, and cell probability threshold.
- **Foci Classification** — Threshold-based morphological pipeline (NLM Denoising, Top-Hat, CLAHE, Area Filtering) to detect foci per cell and classify as Healthy or Damaged.

## Usage

1. Upload a microscopy image (TIF, PNG, JPG).
2. Adjust segmentation parameters and click **Run Segmentation**.
3. Switch to the **Foci Classification** tab, tune parameters, and click **Run Classification**.

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```
