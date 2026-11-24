# PlantDoc AI 🌿
[🚀 Live Demo](https://plantdoc-ai-zulmw2exscqtmk6lyxxqrd.streamlit.app/) | [💻 GitHub Repository](https://github.com/yashasnot/PlantDoc-AI)

**An AI-powered plant disease analysis system** integrating deep learning-based disease classification, visual explainability, leaf segmentation, lesion localization, severity estimation, and prediction reliability assessment.

## Features

- **Disease Classification:** EfficientNet-B0 model with 38 output classes.
- **Explainable AI:** Grad-CAM heatmaps to visualize model activation.
- **Leaf Segmentation:** Isolation of leaf regions from input images.
- **Lesion Localization:** Image-processing-based estimation of lesion regions.
- **Severity Estimation:** Approximate assessment of affected leaf area.
- **Reliability Assessment:** Confidence calibration and prediction stability analysis.
- **Interactive Dashboard:** Streamlit-based interface for image analysis and visual reports.

## Technology Stack

Python · PyTorch · Torchvision · EfficientNet-B0 · Streamlit · OpenCV · NumPy · Pandas · Scikit-learn · Matplotlib

## Project Structure

```text
PlantDoc-AI/
├── app/
│   └── app.py
├── models/
│   └── classification/
├── outputs/
│   ├── heatmaps/
│   ├── masks/
│   └── predictions/
├── reports/
├── scripts/
├── src/
│   ├── explainability/
│   ├── reliability/
│   └── segmentation/
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd plant_disease_final
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app/app.py
```

Open the local URL displayed in your terminal.

## Experimental Results

### Local 15-Class Validation Evaluation

The local validation evaluation contains 3,089 images representing 15 classes. The classifier retains its original 38-class output space.

| Metric | Result |
|---|---:|
| Accuracy | 95.14% |
| Macro F1-score | 0.9534 |
| Weighted F1-score | 0.9546 |
| Calibrated NLL | 0.1427 |
| Calibrated ECE | 0.0062 |

### Reliability Assessment

| Decision | Coverage | Accuracy |
|---|---:|---:|
| Supported | 75.43% | 99.61% |
| Provisional | 14.54% | 90.65% |
| Abstain | 10.04% | 68.06% |

Combined accepted predictions achieved 89.96% coverage and 98.16% accuracy on this validation subset.

These are validation results, not independent external test results.

## Limitations

- The local evaluation uses 15 represented classes, while the classifier has 38 output classes.
- Dataset files are not includeds in this repository.
- Grad-CAM visualizations indicate model activation, not causal explanations.
- Lesion localization and severity estimation are image-processing-based approximations and require expert ground-truth validation.
- High confidence and prediction stability do not guarantee correct predictions.
- Performance on field images and unseen environments requires separate evaluation.

## Reproducibility

Evaluation and analysis scripts are available in the `scripts/` directory. Aggregate experimental metrics and visualizations are available in `reports/`.

## License

No license is currently specified. Check the terms of the underlying dataset and pretrained weights before redistribution.