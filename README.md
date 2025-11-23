# THGSL: Temporal Heterogeneous Graph Structure Learning for Satellite-Video Multi-Object Tracking

Official code for **“THGSL: Temporal Heterogeneous Graph Structure Learning for Satellite-Video Multi-Object Tracking”**  

> **Note on datasets**  
> Datasets are **not included** in this repository because many are large and/or license-restricted.  
> Please **download from the official sources below** and place them under `data/<dataset_name>/`.

---

## Computing Infrastructure <a name="infrastructure"></a>

The experiments reported in the paper were run on the following stack.  

| Component | Specification |
|---|---|
| **CPU** | 2 × Intel Xeon Gold 6430 (32 cores each, 2.8 GHz) |
| **GPU** | 2 × NVIDIA A100 80 GB |
| **System Memory** | 512 GB DDR4-3200 |
| **Storage** | 2 TB NVMe SSD (Samsung PM9A3) |
| **Operating System** | Ubuntu 22.04.4 LTS, Linux 5.15 |
| **CUDA Driver** | 12.1 |
| **cuDNN** | 9.0 |
| **Python Environment** | Conda 23.7 |
| **Other Libraries** | GCC 11.4, CMake 3.29, OpenMPI 4.1 |

---

## Getting Started

### 1) Create a Conda environment (Python 3.8)
```bash
conda create --name THGSL python=3.8
conda activate THGSL
````

### 2) Install non-PyTorch dependencies

```bash
pip install -r requirements.txt
```

### 3) Pin NumPy version

Ensure `numpy==1.24.1` is installed (reinstall if needed):

```bash
pip install --upgrade --force-reinstall numpy==1.24.1
```

### 4) Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

### 5) Install PyTorch Geometric and extensions

```bash
pip install torch-geometric==2.6.1
pip install torch-cluster==1.6.3+pt24cu121 \
            torch-scatter==2.1.2+pt24cu121 \
            torch-sparse==0.6.18+pt24cu121 \
            torch-spline-conv==1.2.2+pt24cu121
```

## Quick Start

```bash
GSL_CAUSAL=1 GSL_ALLOW_INTRA=1 GSL_ALLOW_BACKWARD=0  USE_REAL_GNN=1 GSL_TOPK=50 GSL_TAU=0.2 W_ASSOC=1.0 G2M_L2=1e-4 USE_GNN_FUSION=1 USE_REAL_GNN=1 GATE_FREEZE_EPOCHS=0 python train.py --model_name DLADCN --gpus 0 --lr 3e-5 --lr_step 14 --num_epochs 3000  --batch_size 16 --seqLen 5 --datasetname ICPR --data_dir ./data/ICPR/
```
---

## Datasets (direct links)

> Download from the links below and place files under `data/<dataset_name>/`.
> We do not mirror or redistribute third-party datasets.

### The ICPR SatVideoDT dataset

* Data files:
  https://satvideodt.github.io

### The SatMTB dataset 

* Data files:
  https://zenodo.org/records/15253996

### The SkySat dataset

* Data files:
  https://developers.google.com/earth-engine/datasets/catalog/SKYSAT_GEN-A_PUBLIC_ORTHO_RGB

### The AIR-MOT dataset

* Data files:
  https://drive.google.com/file/d/1xPyTejb7vtZgJzWTFJHBMEW_B7rybid8/view?usp=drive_link


## Notes

* This is an initial, scoped release. We welcome reports of inefficiencies or minor issues.
* Features not included here are planned to be released after acceptance.



## License

Code is released under the MIT License.
Datasets are subject to their respective original licenses/terms.
