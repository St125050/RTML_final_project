# RTML_final_project

Multimodal Vision-Language Models: A Comparative Study of CLIP, BLIP, and ViLT
Project Overview
This repository contains the implementation of a comparative study of three prominent multimodal vision-language models: CLIP, BLIP, and ViLT, as proposed in the research paper by Aakash Kuragayala, Harisa Faiza Dudekula, and Lakshika Padmamali Mahipala Mudiyanselae. The study evaluates these models for zero-shot image classification on the COCO dataset, focusing on performance, robustness, interpretability, and fairness. The project also includes from-scratch implementations of these models to explore custom architectures.
The implementation is provided in a Jupyter Notebook (RTML_fina_project.ipynb), which includes data preparation, model training, and evaluation pipelines. The study uses a subset of 1,000 images from the COCO 2017 dataset and evaluates models under various perturbations (noise, blur, occlusion) and fairness metrics across demographic subgroups.
Objectives
The project aims to:

Evaluate the performance of CLIP, BLIP, and ViLT on zero-shot image classification using the COCO dataset.
Assess robustness by testing models under Gaussian noise, blur, and occlusion perturbations.
Analyze interpretability through attention heatmaps and entropy metrics.
Investigate fairness by evaluating model performance across demographic subgroups.
Compare pre-trained models with custom from-scratch implementations to understand architectural trade-offs.
Provide actionable insights for deploying vision-language models in resource-constrained environments (e.g., Google Colab).

Repository Structure

RTML_fina_project.ipynb: Main Jupyter Notebook containing the complete pipeline for data preparation, model training, and evaluation.
coco_dataset.csv: Generated dataset file with image paths and captions for 1,000 COCO images.
coco_metadata.csv: Metadata file with subgroup annotations for fairness analysis (not provided in the notebook; must be created separately).
Model Weights (generated during training):
clip_model.pth: Trained CLIP model weights.
blip_model.pth, vilt_model.pth, clip_from_scratch.pth, blip_from_scratch.pth, vilt_from_scratch.pth: Weights for other models (if trained).


Output Files:
robustness_interpretability_fairness_results.csv: Results of robustness, interpretability, and fairness evaluations.
<model_name>_attention_maps.png: Attention heatmap visualizations for interpretability.
<model_name>_fairness_plot.png: Bar plots of fairness metrics across subgroups.



Installation
To run the project, set up a Python environment with the required dependencies. The notebook was designed to run on Google Colab (free or Pro tier) with GPU support.
Prerequisites

Python 3.8+
CUDA-enabled GPU (recommended for training and evaluation)
Google Colab or a local environment with sufficient computational resources

Dependencies
Install the required packages using the following command:
pip install torch torchvision transformers pycocotools datasets pandas numpy matplotlib seaborn opencv-python tqdm scipy

Alternatively, use the provided requirements.txt (if available) or install dependencies directly in the notebook using !pip commands.
Setup

Clone the repository:
git clone https://github.com/<your-username>/multimodal-vision-language-models.git
cd multimodal-vision-language-models


Ensure the COCO dataset annotations and images are downloaded (handled automatically by the notebook):

Annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
Images: Downloaded from COCO URLs during execution.


Prepare the coco_metadata.csv file for fairness analysis (not provided in the notebook). This file should contain columns like image_path and category (e.g., gender, ethnicity).


Usage

Open the Notebook:

Upload RTML_fina_project.ipynb to Google Colab or run it locally in Jupyter.
Ensure a GPU runtime is enabled in Colab for faster training and evaluation.


Run the Notebook:

Execute cells sequentially to:
Download and prepare the COCO dataset subset (coco_dataset.csv).
Train the CLIP model (or load pre-trained weights for other models).
Evaluate models for robustness, interpretability, and fairness.
Remove corrupted images (e.g., 000000365426.jpg).


The notebook generates output files (CSV, PNGs) in the working directory.


Key Outputs:

Dataset: coco_dataset.csv with 1,000 image-caption pairs.
Model Weights: Saved as .pth files after training.
Results: robustness_interpretability_fairness_results.csv with metrics for all models.
Visualizations: Attention heatmaps and fairness plots for each model.



Methodology
Data Preparation

Dataset: A subset of 1,000 images from the COCO 2017 dataset with captions.
Processing: Images are resized to 224x224, normalized, and paired with tokenized captions using BertTokenizer or model-specific processors.

Models

Pre-trained Models:
CLIP: Uses ViTModel (DeiT-Small) and DistilBertModel with contrastive loss.
BLIP: Uses BlipForImageTextRetrieval for vision-language alignment.
ViLT: Uses ViltModel with a single-stream transformer for efficiency.


From-Scratch Models:
Custom ViTEncoder and TextEncoder with multi-head attention.
Implementations mimic CLIP, BLIP, and ViLT architectures but with simpler designs (e.g., 256-dimensional embeddings).



Training

CLIP: Trained for 5 epochs with batch size 2, Adam optimizer (lr=1e-4), and gradient accumulation.
Other Models: Evaluated with pre-trained weights or untrained if weights are missing.

Evaluation

Robustness:
Perturbations: Gaussian noise (std: 0.01, 0.05, 0.1), Gaussian blur (kernel: 1, 3, 5), occlusion (patch: 32, 64).
Metrics: Top-1 accuracy, error, average similarity.


Interpretability:
Tools: Attention heatmaps, entropy of attention distributions.
Visualizations: Heatmaps saved as PNGs.


Fairness:
Metrics: Top-1 accuracy and similarity across subgroups (requires coco_metadata.csv).
Visualizations: Bar plots of subgroup performance.



Results

Performance: CLIP is expected to excel in zero-shot classification, BLIP in generative tasks, and ViLT in computational efficiency.
Robustness: Models show varying resilience to noise, blur, and occlusion, with detailed metrics in robustness_interpretability_fairness_results.csv.
Interpretability: Attention heatmaps reveal model focus areas, though ViLT has patch mismatch issues (81 vs. 196 patches).
Fairness: Subgroup performance varies, highlighting potential biases (dependent on coco_metadata.csv).

Limitations

Dataset Size: Only 1,000 COCO images are used, which may limit model generalization.
ViLT Patch Issue: ViLT's patch count (81) mismatches the expected 196, affecting interpretability visualizations.
Missing Metadata: coco_metadata.csv is required for fairness analysis but not provided.
Training Scope: Only CLIP is trained; other models rely on pre-trained or untrained weights.
Resource Constraints: Small batch sizes and GPU memory limitations may affect training efficiency.

Future Work

Expand Dataset: Use a larger COCO subset or include a custom dataset (30–50 images) as proposed.
Fix ViLT: Adjust ViLT's configuration to handle 196 patches consistently.
Generate Metadata: Create coco_metadata.csv with subgroup annotations for fairness analysis.
Train All Models: Train BLIP, ViLT, and from-scratch models to ensure fair comparisons.
Additional Perturbations: Test rotation, color jitter, or text prompt variations for robustness.
Optimize Performance: Use gradient checkpointing or mixed precision to support larger batch sizes.

References

Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020.
Li, J., et al. (2022). BLIP: Bootstrapping Language-Image Pretraining. arXiv:2201.12086.
Kim, W., et al. (2021). ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision. arXiv:2102.03334.
Additional references in the project proposal.

Acknowledgments
This project was developed as part of the Department of Information and Communication Technology at the Asian Institute of Technology, Thailand. The authors thank the academic community and open-source contributors for providing tools like Hugging Face, PyTorch, and the COCO dataset.
Contact
For questions or collaboration, contact:

Aakash Kuragayala: st125050@ait.asia
Harisa Faiza Dudekula: st125053@ait.asia
Lakshika Padmamali Mahipala Mudiyanselae: st124872@ait.asia

