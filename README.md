# Colon Cancer Classification with Explainable AI

This repository presents a deep learning–based approach for classifying
colonoscopy images into three classes: **colon cancer**, **normal colon**,
and **polyp**.

Two transfer learning architectures, **InceptionV3** and **Xception**, are
used and compared. Model predictions are interpreted using **Grad-CAM**
to provide visual explanations highlighting clinically relevant regions
in the images.

The repository focuses on model architecture, training methodology,
evaluation results, and explainable AI visualizations.

## Results

Both InceptionV3 and Xception models were trained using transfer learning on
colonoscopy image data. Training and validation performance is visualized
using accuracy and loss curves, along with confusion matrices for
class-wise evaluation.

Xception achieved higher validation accuracy and more stable convergence
compared to InceptionV3. Confusion matrices indicate that most
misclassifications occur between colon cancer and normal colon classes,
which is expected due to visual similarity in endoscopic imagery.

Explainable AI analysis was performed using Grad-CAM. The resulting heatmaps
demonstrate that both models focus on clinically relevant regions such as
lesion boundaries and abnormal tissue patterns. XAI visualizations are
provided in the `outputs/` directory.
