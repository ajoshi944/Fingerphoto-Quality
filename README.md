# UFQA: Utility Guided Fingerphoto Quality Assessment

This repository contains the official implementation of the paper:  
**"UFQA: Utility Guided Fingerphoto Quality Assessment"**  
*Amol S. Joshi, Ali Dabouei, Jeremy Dawson, Nasser Nasrabadi*  
[📄 Paper Link](https://arxiv.org/abs/2407.11141)

UFQA is a self-supervised dual encoder framework for predicting fingerphoto quality scores aligned with biometric matching performance. The model leverages quality maps for additional supervision, ensuring both global utility and local image quality are accurately captured.

---

## 📌 Abstract
Quality assessment of fingerphotos i.e. fingerprints captured using smartphone or digital cameras, is essential for reliable biometric recognition. UFQA uses a self-supervised dual encoder architecture to fuse features in latent space, predicting a quality metric aligned with utility in matching scenarios. Experiments show UFQA outperforms NFIQ2.2 and other state-of-the-art image quality assessment algorithms on multiple public datasets.

---

## 🏗 Architecture
![UFQA Architecture](images/UFQA_architecture.png)  
*UFQA architecture: dual encoders with quality map supervision.*

---

## ✨ Features
- **Self-supervised dual encoder** for latent feature fusion.
- **Quality map supervision** for robustness to local distortions.
- Predicts **utility-aligned quality scores** for biometric matching.
- Outperforms **NFIQ2.2** and other metrics on multiple datasets.

---

## ⚙️ Installation
```bash
git clone https://github.com/ajoshi944/Fingerphoto-Quality.git
cd UFQA
pip install -r requirements.txt
```
