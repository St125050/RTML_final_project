# RTML_final_project
# Multimodal Vision-Language Models: A Comparative Study of CLIP, BLIP, and ViLT

## Project Overview
This repository contains the implementation of a comparative study of three prominent multimodal vision-language models: CLIP, BLIP, and ViLT, as proposed in the research paper by Aakash Kuragayala, Harisa Faiza Dudekula, and Lakshika Padmamali Mahipala Mudiyanselae. The study evaluates these models for zero-shot image classification on the COCO dataset, focusing on performance, robustness, interpretability, and fairness. The project also includes from-scratch implementations of these models to explore custom architectures.

The implementation is provided in a Jupyter Notebook (`RTML_fina_project.ipynb`), which includes data preparation, model training, and evaluation pipelines. The study uses a subset of 1,000 images from the COCO 2017 dataset and evaluates models under various perturbations (noise, blur, occlusion) and fairness metrics across demographic subgroups.

---

## Objectives
The project aims to:
- Evaluate the performance of CLIP, BLIP, and ViLT on zero-shot image classification using the COCO dataset.
- Assess robustness by testing models under Gaussian noise, blur, and occlusion perturbations.
- Analyze interpretability through attention heatmaps and entropy metrics.
- Investigate fairness by evaluating model performance across demographic subgroups.
- Compare pre-trained models with custom from-scratch implementations to understand architectural trade-offs.
- Provide actionable insights for deploying vision-language models in resource-constrained environments (e.g., Google Colab).

---

## Repository Structure
- **RTML_fina_project.ipynb**: Main Jupyter Notebook containing the complete pipeline for data preparation, model training, and evaluation.
- **coco_dataset.csv**: Generated dataset file with image paths and captions for 1,000 COCO images.
- **coco_metadata.csv**: Metadata file with subgroup annotations for fairness analysis (not provided in the notebook; must be created separately).
- **Model Weights (generated during training)**:
  - `clip_model.pth`: Trained CLIP model weights.
  - `blip_model.pth`, `vilt_model.pth`, `clip_from_scratch.pth`, `blip_from_scratch.pth`, `vilt_from_scratch.pth`: Weights for other models (if trained).
- **Output Files**:
  - `robustness_interpretability_fairness_results.csv`: Results of robustness, interpretability, and fairness evaluations.
  - `<model_name>_attention_maps.png`: Attention heatmap visualizations for interpretability.
  - `<model_name>_fairness_plot.png`: Bar plots of fairness metrics across subgroups.

---

## Installation
To run the project, set up a Python environment with the required dependencies. The notebook was designed to run on Google Colab (free or Pro tier) with GPU support.

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended for training and evaluation)
- Google Colab or a local environment with sufficient computational resources

### Dependencies
Install the required packages using the following command:
```bash
pip install torch torchvision transformers pycocotools datasets pandas numpy matplotlib seaborn opencv-python tqdm scipy
```

Alternatively, use the provided `requirements.txt` (if available) or install dependencies directly in the notebook using `!pip` commands.

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/multimodal-vision-language-models.git
   cd multimodal-vision-language-models
   ```

2. Ensure the COCO dataset annotations and images are downloaded (handled automatically by the notebook):
   - **Annotations**: [Download annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
   - **Images**: Downloaded from COCO URLs during execution.

3. Prepare the `coco_metadata.csv` file for fairness analysis (not provided in the notebook). This file should contain columns like `image_path` and `category` (e.g., gender, ethnicity).

---

## Usage

### Open the Notebook:
- Upload `RTML_fina_project.ipynb` to Google Colab or run it locally in Jupyter.
- Ensure a GPU runtime is enabled in Colab for faster training and evaluation.

### Run the Notebook:
Execute cells sequentially to:
1. Download and prepare the COCO dataset subset (`coco_dataset.csv`).
2. Train the CLIP model (or load pre-trained weights for other models).
3. Evaluate models for robustness, interpretability, and fairness.
4. Remove corrupted images (e.g., `000000365426.jpg`).

The notebook generates output files (CSV, PNGs) in the working directory.

---

## Key Outputs
- **Dataset**: `coco_dataset.csv` with 1,000 image-caption pairs.
- **Model Weights**: Saved as `.pth` files after training.
- **Results**: `robustness_interpretability_fairness_results.csv` with metrics for all models.
- **Visualizations**: Attention heatmaps and fairness plots for each model.

---

## Methodology

### Data Preparation
- **Dataset**: A subset of 1,000 images from the COCO 2017 dataset with captions.
- **Processing**: Images are resized to 224x224, normalized, and paired with tokenized captions using `BertTokenizer` or model-specific processors.

### Models
- **Pre-trained Models**:
  - **CLIP**: Uses `ViTModel` (DeiT-Small) and `DistilBertModel` with contrastive loss.
  - **BLIP**: Uses `BlipForImageTextRetrieval` for vision-language alignment.
  - **ViLT**: Uses `ViltModel` with a single-stream transformer for efficiency.
- **From-Scratch Models**:
  - Custom `ViTEncoder` and `TextEncoder` with multi-head attention.
  - Implementations mimic CLIP, BLIP, and ViLT architectures but with simpler designs (e.g., 256-dimensional embeddings).

### Training
- **CLIP**: Trained for 5 epochs with batch size 2, Adam optimizer (`lr=1e-4`), and gradient accumulation.
- **Other Models**: Evaluated with pre-trained weights or untrained if weights are missing.

### Evaluation
1. **Robustness**:
   - **Perturbations**: Gaussian noise (std: 0.01, 0.05, 0.1), Gaussian blur (kernel: 1, 3, 5), occlusion (patch: 32, 64).
   - **Metrics**: Top-1 accuracy, error, average similarity.
2. **Interpretability**:
   - **Tools**: Attention heatmaps, entropy of attention distributions.
   - **Visualizations**: Heatmaps saved as PNGs.
3. **Fairness**:
   - **Metrics**: Top-1 accuracy and similarity across subgroups (requires `coco_metadata.csv`).
   - **Visualizations**: Bar plots of subgroup performance.

---

## Analysis of Visualizations

### Attention Heatmaps (Interpretability)
The heatmaps (Samples 1 to 5) show the attention weights of a model (likely CLIP, BLIP, or ViLT) on 224x224 images divided into patches (14x14 grid, 196 patches total). The color scale ranges from ~0.0040 (dark blue, low attention) to ~0.070 (yellow, high attention).

**Observations**:
1. **Sample 1**: High attention (yellow) in patches around `(0, 0)` and `(10, 10)`, suggesting the model focuses on distinct image regions (possibly objects or key features).
2. **Samples 2 and 4**: More distributed attention with multiple patches having moderate attention (green, ~0.050–0.060), indicating the model might be focusing on broader areas or struggling to pinpoint a single region.
3. **ViLT Patch Mismatch Issue**: The mismatch (81 patches vs. expected 196) likely affects these heatmaps, suggesting the heatmaps might be from CLIP or BLIP, or the ViLT issue was resolved for visualization.

### Fairness Bar Plots
The bar plots show Top-1 accuracy across COCO categories (subgroups) for CLIP, BLIP, ViLT, and their from-scratch variants.

**CLIP Fairness**:
- Accuracy varies significantly across categories, ranging from ~0.5 to 2.5.
- Categories like "fire hydrant" and "tennis racket" have high accuracy (~2.5), while others like "carrot" and "teddy bear" are lower (~0.5).
- Suggests CLIP may be biased toward certain object types, possibly due to imbalanced training data.

**BLIP Fairness**:
- Accuracy ranges from ~0.5 to 3.0, with "fire hydrant" peaking at ~3.0.
- BLIP seems to perform better on some categories compared to CLIP, possibly due to its generative pretraining approach.

**ViLT Fairness**:
- Accuracy is more uniform, ranging from ~6.5 to 7.5 across all categories.
- Suggests less bias across subgroups but lower overall performance compared to CLIP/BLIP.

**From-Scratch Models**:
- Show lower accuracy (0.2 to 1.2) compared to pre-trained counterparts, with more variability across categories.
- ViLT's from-scratch variant does not degrade as much, possibly due to its simpler architecture.

---

## Results

1. **Performance Overview**:
   - **CLIP**: Top-1 accuracy is 0.1955, with a negative avg_similarity (-0.034). Top-5 accuracy reaches 0.74.
   - **BLIP**: Slightly better Top-1 accuracy (0.2125) and positive avg_similarity (0.161), with Top-5 accuracy at 0.7855.
   - **ViLT and ViLTFromScratch**: Both show perfect scores (1.0), indicating a likely evaluation error.
   - **From-Scratch Models**: Lower accuracy compared to pre-trained counterparts.

2. **Robustness**:
   - **CLIP and BLIP**: Accuracy drops with noise and blur but improves slightly under larger occlusion.
   - **ViLT and From-Scratch Models**: Exhibit uniform performance, likely due to evaluation issues.

3. **Interpretability**:
   - **Attention Entropy**:
     - CLIP: 3.872 (focused attention).
     - BLIP: 5.277 (distributed attention).
     - From-scratch models: Higher entropy (~5.2).

---

## Limitations
1. **Dataset Size**: Only 1,000 COCO images are used.
2. **ViLT Patch Issue**: Needs fixed patch count.
3. **Expand Dataset**: Include more images for better generalization.

---

## Contact
For questions or collaboration, contact:
- **Aakash Kuragayala**: [st125050@ait.asia](mailto:st125050@ait.asia)
- **Harisa Faiza Dudekula**: [st125053@ait.asia](mailto:st125053@ait.asia)
- **Lakshika Padmamali Mahipala Mudiyanselae**: [st124872@ait.asia](mailto:st124872@ait.asia)
