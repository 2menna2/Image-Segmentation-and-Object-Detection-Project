# Image Segmentation and Object Detection Project

This project performs image segmentation and object detection using two approaches: U-Net with a ResNet34 encoder for segmentation and a simplified YOLOv1 model for object detection. It processes HEIC images, converts them to JPEG, generates segmentation masks, splits the dataset into train/validation/test sets, and trains both models for segmentation and detection tasks.
## Directory Structure
project/
│
├── data/
│   └── *.heic / *.jpg
├── masks/
│   └── *.png
├── split/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   └── test/
├── yolo_labels/
│   ├── train/
│   └── val/
├── notebooks/  # Optional for Colab notebooks
├── convert_heic_to_jpg.py
├── generate_masks.py
├── split_dataset.py
├── train_unet.py
├── masks_to_yolo_labels.py
├── train_yolov1.py
└── visualize_predictions.py

## Project Structure
- **Data Preparation**:
  - Converts HEIC images to JPEG using `pillow_heif`.
  - Generates binary segmentation masks from JSON annotations.
  - Splits dataset into train (70%), validation (20%), and test (10%) sets.
- **Segmentation**:
  - Uses U-Net with ResNet34 encoder from `segmentation_models_pytorch`.
  - Applies data augmentation with `albumentations`.
  - Trains with combined Dice and BCE loss.
- **Object Detection**:
  - Implements a simplified YOLOv1 model for bounding box detection.
  - Converts masks to YOLO-format bounding box labels.
  - Trains with a custom YOLO loss function.

## Requirements
See `requirements.txt` for the list of required Python packages.

## Setup
1. **Mount Google Drive**:
   - The project assumes data is stored in Google Drive under `/content/drive/MyDrive/annotati`.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Directory Structure**:
   - Input images: `/content/drive/MyDrive/annotati/data`
   - JSON annotations: `/content/drive/MyDrive/annotati/labels_my-project-name_2025-09-18-04-55-28.json`
   - Output masks: `/content/drive/MyDrive/annotati/masks`
   - Split dataset: `/content/drive/MyDrive/annotati_split`
   - YOLO labels: `/content/yolo_labels`

## Usage
1. **Convert HEIC to JPEG**:
   - Run the HEIC-to-JPEG conversion script to process images in `src_dir` and save them to `dst_dir`.
2. **Generate Masks**:
   - Process JSON annotations to create binary masks for each image.
3. **Split Dataset**:
   - Split images and masks into train, validation, and test sets.
4. **Train U-Net Model**:
   - Train the U-Net model for segmentation with specified hyperparameters (e.g., 15 epochs, batch size 4).
   - Save the best model to `/content/drive/MyDrive/annotati/best_unet_resnet34.pth`.
5. **Train YOLOv1 Model**:
   - Generate YOLO-format labels and train the YOLOv1 model for 10 epochs.
   - Save the best model to `/content/best_yolov1.pth`.
6. **Inference**:
   - Visualize segmentation results on a random validation image using U-Net.
   - Visualize bounding box predictions on a random validation image using YOLOv1.

## Results
- **Segmentation**: Displays original image, ground truth mask, and predicted mask.
- **Object Detection**: Displays bounding boxes with confidence scores on a random validation image.

## Notes
- Ensure sufficient Google Drive space for storing images, masks, and models.
- Adjust hyperparameters (e.g., `img_size`, `batch_size`, `epochs`) based on your dataset and hardware.
- The project uses `cuda` if available; otherwise, it falls back to CPU.