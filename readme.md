# Face Aging with CycleGAN (UTKFace)

This project implements a **face aging model using CycleGAN** trained on a subset of the **UTKFace dataset**.
The model learns to translate faces between **young and old domains** using **unpaired image-to-image translation**.

The project was created to explore how generative adversarial networks can simulate **age progression on human faces**.

---

# Project Overview

The model is based on the **CycleGAN architecture**, which allows training without paired images.
Instead of needing the same person at different ages, the model learns the transformation between two domains:

* **Young faces**
* **Old faces**

Two generators learn opposite mappings while cycle consistency ensures that the identity of the face is preserved.

---

# Dataset

The model was trained on the **UTKFace dataset**.

Dataset preparation:

* Images resized to **128 × 128**
* Approximately **3000 images used for training**
* Faces divided into two domains based on age labels:

  * **Young**
  * **Old**

UTKFace contains variations in:

* age
* gender
* ethnicity
* lighting
* pose

These variations help the model learn general aging patterns.

---

# Model Architecture

The implementation follows the standard **CycleGAN architecture**.

## Generators

Two generators are used:

* **Generator G:** Young → Old
* **Generator F:** Old → Young

Each generator contains:

* convolutional layers
* residual blocks
* instance normalization
* upsampling layers

## Discriminators

Two **PatchGAN discriminators** are used:

* **Discriminator X** — distinguishes real vs generated young faces
* **Discriminator Y** — distinguishes real vs generated old faces

PatchGAN focuses on local image patches which helps preserve texture details.

---

# Training Configuration

| Parameter       | Value        |
| --------------- | ------------ |
| Image size      | 128 × 128    |
| Training images | ~3000        |
| Batch size      | 1–2          |
| Optimizer       | Adam         |
| Learning rate   | 0.0002       |
| Epochs          | ~220         |

### Loss Functions

The training objective includes:

* **Adversarial loss** – makes generated images realistic
* **Cycle consistency loss** – preserves facial identity
* **Identity loss** – stabilizes training and preserves color structure

---

# Training Pipeline

Young image:

```
Young → Generator G → Fake Old
Fake Old → Generator F → Reconstructed Young
```

Old image:

```
Old → Generator F → Fake Young
Fake Young → Generator G → Reconstructed Old
```

Cycle consistency ensures:

```
Young → Old → Young ≈ original image
Old → Young → Old ≈ original image
```

---

# Inference Pipeline

```
Input image
     ↓
Preprocessing (resize to 128×128)
     ↓
CycleGAN Generator (Young → Old)
     ↓
Generated aged face
```

Optional post-processing can include:

* face restoration
* super-resolution

---

# Project Structure

```
face-aging-cyclegan
│
├── dataset
│   ├── young
│   └── old
│
├── models
│   ├── generator.py
│   ├── discriminator.py
│
├── training
│   └── train.py
│
├── inference
│   └── generate.py
│
├── checkpoints
│
└── README.md
```

---

# Installation

Clone the repository:

```
git clone https://github.com/your-username/face-aging-cyclegan.git
cd face-aging-cyclegan
```

Install dependencies:

```
pip install -r requirements.txt
```

Main dependencies:

* Python 3.10+
* TensorFlow / PyTorch
* NumPy
* OpenCV
* Matplotlib

---

# Results

The model learns several aging characteristics:

* wrinkles
* skin texture changes
* subtle facial structure shifts

Because the training dataset is relatively small (~3000 images), the results may sometimes include:

* blur
* artifacts
* limited realism

Increasing dataset size or resolution can significantly improve results.

---

# Possible Improvements

Future improvements could include:

* training on **larger UTKFace subsets (20k+ images)**
* increasing resolution to **256×256**
* integrating **face restoration models**
* adding **super-resolution post-processing**
* experimenting with **diffusion-based aging models**

---

# References

CycleGAN paper
https://arxiv.org/abs/1703.10593

UTKFace dataset
https://susanqq.github.io/UTKFace/

---

# License

This project is intended for **educational and research purposes**.
