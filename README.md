# iGAN: Interactive Image Generation via GAN

> A deep learning-powered interior design assistant that converts user-drawn sketches into photorealistic images using Pix2Pix GANs.

## 📌 Project Overview

**iGAN** is an interactive application that uses Conditional Generative Adversarial Networks (cGANs), specifically the Pix2Pix model, to translate edge-detected room sketches into high-quality interior design images. The tool is designed for quick visualization and design iteration for home office spaces—allowing even non-experts to sketch, generate, and refine room layouts in real-time.

---

## 🧠 Core Technologies and Libraries

| Category | Library / Tool | Purpose |
|---------|----------------|---------|
| **Language** | Python | Core development |
| **Deep Learning** | TensorFlow / Keras / PyTorch | Model training and inference |
| **GAN Framework** | Pix2Pix (U-Net + PatchGAN) | Image-to-image translation |
| **Image Processing** | OpenCV | Edge detection via Canny |
| **UI Framework** | PyQt5 / Tkinter | Desktop GUI for sketching and visualization |
| **Web Integration** | Flask (optional) | Backend-frontend model serving |
| **Visualization** | Matplotlib / Seaborn | Model evaluation and metrics |
| **Hardware Acceleration** | CUDA | GPU-based training/inference |

---

## 🎯 Features

- ✍️ **Sketch Interface**: Users can draw room layouts via PyQt canvas or upload sketches.
- ⚙️ **Real-Time Edge Detection**: Canny edge detection converts input to edge maps.
- 🎨 **GAN-Generated Output**: Pix2Pix model generates realistic interior visuals.
- 📈 **Performance Metrics**: Evaluation via SSIM, PSNR, and adversarial loss.
- 🧪 **Testing Suite**: Structured and unstructured input testing, model validation.
- 🖥️ **Desktop & Web Deployment**: Local GUI and optional cloud/web support.

---

## 🧱 Pix2Pix GAN Architecture

### Generator
- **Type**: U-Net (Encoder–Decoder with skip connections)
- **Input**: Edge-detected sketch (256x256)
- **Output**: Photorealistic RGB image

### Discriminator
- **Type**: PatchGAN (70x70 receptive fields)
- **Purpose**: Classifies real vs. fake image patches to ensure local realism

### Loss Functions
- **L1 Loss**: Ensures pixel-wise accuracy
- **Adversarial Loss**: Encourages realism via GAN training

---

## 🧰 Dataset

- **Type**: Paired dataset (Edge Map + Real Image)
- **Source**: Custom + public datasets (e.g., Places365)
- **Preprocessing**:
  - Resizing to 256x256
  - Grayscale conversion
  - Canny edge detection
- **Augmentation**:
  - Rotation, flipping, brightness adjustments

---

## 🖼️ Example Workflow

1. User draws/sketches a room layout
2. The image is converted into an edge map using Canny detection
3. The edge map is input into the trained Pix2Pix GAN
4. The GAN outputs a realistic render of the interior design

---

## 📊 Evaluation Metrics

- **SSIM (Structural Similarity Index)**
- **PSNR (Peak Signal-to-Noise Ratio)**
- **Inception Score**
- **Qualitative Testing**: Human evaluation and feedback

---

## 🧪 Testing Strategy

- **Structured Inputs**: Template-based room sketches
- **Unstructured Inputs**: Freehand user sketches
- **Overfitting Monitoring**: Via validation loss
- **User Feedback**: Visual realism, ease of use, output relevance

---

## 🚀 Deployment

### Local Desktop
- **Built With**: PyQt5 or Tkinter
- **Runs On**: Windows/Linux with Python

### Optional Web App
- **Framework**: Flask backend
- **Model Format**: TensorFlow.js or ONNX for browser compatibility
- **GPU Acceleration**: CUDA-enabled inference

---

## 🔮 Future Scope

- 🧱 3D Room Rendering and Walkthroughs
- 🧠 Semantic Sketch Labeling (e.g., "chair", "window")
- ☁️ Cloud + Mobile Integration
- 🪑 Smart Furniture Arrangement via ML

---

## 👩‍💻 Contributors

- **Vibhav Chhabra** – 21CSU347  
- **Shivam Bhardwaj** – 21CSU325  
- **Nilesh Tyagi** – 21CSU307  
- **Sakshi** – 21CSU419  

**Supervisor**: Dr. Tamalika Chaira  
**Institution**: The NorthCap University, Gurugram

---

## 📚 References

See [References](#) section in the project PDF for detailed citations on Pix2Pix, GANPaint Studio, RoomGAN, iGAN, DragGAN, and more.

---

## 📂 Project Structure (Sample)

