<div align="center">
<h1>
<b>
ChangeUNet: Change Detection with ResNet18 + UNet
</b>
</h1>
</div>

![Demo Image](https://github.com/KennyChen880127/ChangeUNet/blob/main/assets/Demo.png?raw=true)

# Introduction
This project implements **Change-UNet**, a change detection architecture designed to identify differences between bi-temporal remote sensing images.  
The model combines a **ResNet18 encoder** for feature extraction, a **FuseReduce module** to integrate features from two time points (A and B), and a **U-Net style decoder** to generate high-resolution change masks.

- **Encoder**: ResNet18 backbone to extract multi-scale features (C1–C4).  
- **Feature Fusion**: For each level, features from image A and B are fused with their absolute difference `[fa, fb, |fa-fb|]` and reduced using a 1×1 convolution.  
- **Decoder**: A UNet-like upsampling path with skip connections reconstructs the fused features into a full-resolution mask.  
- **Output Head**: Produces a binary change mask where 1 = changed, 0 = unchanged.  

We evaluate this model on the [LEVIR-CD dataset](https://justchenhao.github.io/LEVIR/), a widely used benchmark for remote sensing change detection.  
LEVIR-CD contains 637 pairs of high-resolution (1024×1024) aerial images annotated with building change masks.  

# Architecture
![Architecture](https://github.com/KennyChen880127/ChangeUNet/blob/main/assets/architecture.jpg?raw=true)

<details>
  <summary><strong>Input Image</strong></summary>

- Shape: (B, 3, 512, 512)  
- B = Batch size  
- 3 = Channels (RGB)  
- 512×512 = Image resolution  

</details>

<details>
  <summary><strong>ResNet18 Encoder (shared for Image A & B)</strong></summary>

- Stem (Conv7×7, stride=2) → (B, 64, 256, 256)  
- MaxPool (3×3, stride=2) → (B, 64, 128, 128)  
- Layer1 → C1: (B, 64, 128, 128)  
- Layer2 → C2: (B, 128, 64, 64)  
- Layer3 → C3: (B, 256, 32, 32)  
- Layer4 → C4: (B, 512, 16, 16)  

</details>

<details>
  <summary><strong>Feature Fusion (FuseReduce)</strong></summary>

- **Definition:**  
  FuseReduce = [fa, fb, |fa - fb|] → Concat → 1×1 Conv → BatchNorm → ReLU  

- **Outputs:**  
  - F1 = FuseReduce(C1A, C1B, |C1A - C1B|) → (B, 64, 128, 128)  
  - F2 = FuseReduce(C2A, C2B, |C2A - C2B|) → (B, 128, 64, 64)  
  - F3 = FuseReduce(C3A, C3B, |C3A - C3B|) → (B, 256, 32, 32)  
  - F4 = FuseReduce(C4A, C4B, |C4A - C4B|) → (B, 512, 16, 16)  

</details>

<details>
  <summary><strong>UNet Decoder</strong></summary>

- **UpBlock definition (basic unit):**  
  1. ConvTranspose2d (upsampling ×2)  
  2. Concatenate with corresponding skip feature (F3/F2/F1) along channel dimension  
  3. DoubleConv: two Conv2d(3×3, padding=1) + BatchNorm + ReLU  

---

- **Decoding process (top-down with skip connections):**  
  - Start: `x = F4`  
  - `x = UpBlock(x, F3)` → (B, 256, H/16, W/16)  
  - `x = UpBlock(x, F2)` → (B, 128, H/8,  W/8)  
  - `x = UpBlock(x, F1)` → (B,  64, H/4,  W/4)  
  - Final: `x = ConvTranspose2d(x)` → (B, 32, H, W)  

</details>

<details>
  <summary><strong>Output Head</strong></summary>

- DoubleConv(32 → 32) (keeps spatial size)  
- Conv2d(1×1, 32 → 1) → logits (B, 1, H, W)  
- Sigmoid(logits) → change mask probability map `(B, 1, H, W)` (threshold > 0.5 → 1)  

</details>