# Isonet

## 도입(Introduction)

This project predicts ADMET properties from SMILES using a **Directed Message-Passing Neural Network (DMPNN)**.

본 프로젝트는 SMILES로부터 분자 그래프를 구성하고, DMPNN을 이용하여 ADMET 물성을 예측하는 연구이다. 입체화학(stereochemistry) feature, 자기지도 사전학습, Stereo Attention이 예측 성능에 미치는 영향을 비교한다.

현재 결과는 **PPBR_AZ(혈장 단백질 결합률) 회귀 예측**을 중심으로 정리하였다. 기존 모델도 입체화학 정보를 사용할 수 있다는 점을 바탕으로, 입체화학 정보를 입력하고 입체중심 주변의 표현을 추가로 강조하는 방식의 효과를 검증한다.

## 연구 가설(Research Hypothesis)

> **H1.** stereochemistry 정보를 atom/bond feature에 포함하면 stereo 정보를 포함하지 않은 모델보다 특히 stereochemical subset에서 ADMET 예측 성능이 향상될 것이다.

> **H2.** stereocenter와 그 주변 local representation을 별도의 attention mechanism으로 집약하면, 단순히 stereo feature만 제공하는 모델보다 stereochemical subset의 ADMET 예측 성능이 향상될 것이다.

추가로, 분자 그래프의 마스킹 복원 과제로 사전학습한 DMPNN이 무작위 초기화한 DMPNN보다 downstream 예측에 유리한지 비교한다. 아래 중간 결과는 전체 test set에 대한 결과이며, stereochemical subset에 대한 가설 검증은 후속 과제이다.

## 모델 구조(Model Architecture)

**SMILES → 분자 그래프 → DMPNN → Stereo Attention(선택) → Pooling → MLP → 물성 예측**

- **분자 그래프:** RDKit으로 원자·결합 feature를 구성한다. 입체화학 feature 사용 여부를 변경할 수 있다.
- **DMPNN:** 방향성 결합을 따라 메시지를 전달하여 원자별 표현을 생성한다.
- **Stereo Attention:** CIP 표지 R/S/r/s가 있는 입체중심을 query로, 직접 연결된 이웃 원자의 표현과 결합 feature를 key/value로 사용한다. 이웃 정보를 attention으로 집약하고 gate를 거쳐 입체중심의 표현에 더한다.
- **Pooling:** 원자별 표현을 Sum 또는 Mean pooling으로 집약하여 분자 표현을 만든다.
- **MLP:** 분자 표현으로부터 endpoint별 값을 예측한다.

Stereo Attention은 DMPNN 이후, pooling 이전에 적용된다. 현재 구현은 CIP 표지가 있는 원자 중심의 attention이며, E/Z 결합을 중심으로 별도의 attention을 수행하지는 않는다.

## 진행 실험(Experiments)

### 실험 1. Stereo feature 효과 비교

입체화학 feature를 사용하는 조건과 사용하지 않는 조건을 비교한다. Sum 및 Mean pooling 각각에서 feature ON/OFF에 따른 예측 성능의 차이를 확인한다.

### 실험 2. 자기지도 사전학습 효과 비교

원자 또는 결합 feature 일부를 마스킹하고, 원자·결합 종류를 복원하도록 DMPNN을 사전학습한다. 사전학습한 encoder로 초기화한 모델과 무작위 초기화한 모델을 각각 fine-tuning하여 비교한다.

### 실험 3. Stereo Attention 효과 비교

입체화학 feature를 사용하는 DMPNN에서 Stereo Attention을 적용한 조건과 적용하지 않은 조건을 비교한다. DMPNN 이후, pooling 이전에 입체중심 주변의 표현을 추가로 집약하는 것이 예측 성능에 미치는 영향을 확인한다.

### 실험 코드와 결과 파일 대응

위 설명의 실험 번호와 저장소의 `exNN` 번호는 다르다.

| 실행 파일 | 목적 | `metrics_1.json` | `metrics_2.json` |
| --- | --- | --- | --- |
| [ex01/main_01.py](isonet/experiment/ex01/main_01.py) | Stereo feature 비교 | ON | OFF |
| [ex02/main_02.py](isonet/experiment/ex02/main_02.py) | 사전학습 비교 | 사전학습 적용 | 무작위 초기화 |
| [ex03/main_03.py](isonet/experiment/ex03/main_03.py) | 공용 encoder 사전학습·학습 재개 | 해당 없음 | 해당 없음 |
| [ex04/main_04.py](isonet/experiment/ex04/main_04.py) | Pooling 비교 | Sum | Mean |
| [ex05/main_05.py](isonet/experiment/ex05/main_05.py) | Stereo Attention 비교 | ON | OFF |

## 중간 결과(Interim Results)

2026-10-21 기준 저장소의 `temp/`에 보관된 `metrics_*.json`을 집계하였다. 각 값은 **학습 시드 3개에 대한 평균 ± 표본 표준편차**이며, 각 실행의 test sample 수는 163개이다. RMSE와 MAE는 낮을수록 좋다. 학습 시드 반복은 학습의 변동성을 확인하기 위한 것이며, 서로 다른 데이터 분할에서의 재현성을 검증한 것으로 해석하지 않는다.

이 표는 저장된 평가 지표의 요약이며, 현재 코드로 모든 과거 실험을 다시 학습하여 재현한 결과는 아니다. 각 비교군 내부의 ON/OFF를 비교하며, 서로 다른 실험군 사이의 차이를 특정 모듈 하나의 효과로 해석하지 않는다.

### 1. Stereo feature

결과: [`temp/stereo_feature/`](temp/stereo_feature/) · 학습 시드: **25, 40, 42**

| Pooling | Stereo feature | RMSE ↓ | MAE ↓ |
| --- | --- | --- | --- |
| Sum | OFF | 14.301 ± 0.215 | 10.793 ± 0.164 |
| Sum | ON | 14.973 ± 0.414 | 11.592 ± 0.501 |
| Mean | OFF | 14.583 ± 0.106 | 9.986 ± 0.170 |
| Mean | ON | 14.029 ± 0.142 | 9.856 ± 0.062 |

Mean pooling에서는 feature 추가 시 평균 RMSE와 MAE가 감소했지만, Sum pooling에서는 증가하였다. 입체화학 feature의 효과는 pooling 조건에 따라 달랐다. Mean pooling 조건의 평균 RMSE는 약 3.8%, MAE는 약 1.3% 감소하였다. 이는 해당 조건에서 feature 추가가 유리했음을 보여 주지만, 이성질체 간 물성 차이를 더 정확히 구분한다는 결론에는 stereo subset 및 이성질체 pair 평가가 필요하다.

### 2. 자기지도 사전학습

결과: [`temp/pretrain_rlt/`](temp/pretrain_rlt/)의 `*seed_re` 폴더 · 학습 시드: **20, 25, 42**

| 사전학습 | RMSE ↓ | MAE ↓ |
| --- | --- | --- |
| OFF | 13.602 ± 0.289 | 9.526 ± 0.377 |
| ON | 13.223 ± 0.200 | 9.329 ± 0.161 |

사전학습 적용 시 평균 RMSE는 약 2.8%, MAE는 약 2.1% 감소하였다. 이는 이번 비교 조건에서 사전학습한 표현으로 초기화하는 방식이 예측에 도움이 되었음을 시사한다. 다만 사전학습의 이득이 입체화학 정보 학습에서 비롯되었는지는 이 비교만으로 판단할 수 없다.

### 3. Stereo Attention

학습 시드: **20, 25, 40**. 비교군은 결과가 저장된 폴더 기준으로 구분하였다.

| 결과 폴더 / 조건 | Attention | RMSE ↓ | MAE ↓ |
| --- | --- | --- | --- |
| `attn_compare` | OFF | 13.693 ± 0.256 | 9.752 ± 0.167 |
| `attn_compare` | ON | 13.912 ± 0.116 | 9.980 ± 0.110 |
| `attn_compare_with_lr/*_add` | OFF | 15.643 ± 0.159 | 11.257 ± 0.197 |
| `attn_compare_with_lr/*_add` | ON | 15.326 ± 0.361 | 11.274 ± 0.502 |
| `attn_compare_with_lr/*_mean` | OFF | 13.906 ± 0.158 | 9.774 ± 0.131 |
| `attn_compare_with_lr/*_mean` | ON | 15.076 ± 0.065 | 10.733 ± 0.211 |

결과: [`temp/attn_compare/`](temp/attn_compare/), [`temp/attn_compare_with_lr/`](temp/attn_compare_with_lr/)

`*_add` 비교군에서는 평균 RMSE가 감소했으나 MAE는 개선되지 않았다. 현재 결과만으로 Stereo Attention의 일관된 성능 향상을 확인하지는 못하였다. RMSE와 MAE의 변화 방향이 다른 이유는 분자별 오차를 확인해야 판단할 수 있다. 현재 결과는 제안한 attention 구성의 추가 이득을 뒷받침하기에 부족하며, 원인 분석에는 stereo subset 평가와 학습·검증 손실, attention 및 gate 값의 확인이 필요하다. 위 평균 차이에 대한 통계적 유의성은 검정하지 않았다.

## 후속 실험 계획(Future Work)

- **Stereo subset 평가:** 전체 ADMET test set뿐 아니라 R/S 또는 E/Z 정보가 존재하는 분자만 따로 추출한 subset에서도 성능을 평가한다.
- **모델 분석:** 입체중심 주변 원자의 attention weight를 분석하고, 실제 데이터에 이성질체 pair가 충분히 있다면 실제 물성 차이와 모델이 예측한 차이를 비교한다.
- **파라미터 ablation study:** `depth`, `hidden_dim`, pooling 및 attention 설정의 영향을 비교한다.
- **외부 모델 비교:** Chemprop(DMPNN), GROVER, MolFormer, Mole-BERT와 동일한 데이터 분할·평가 조건으로 비교한다.
- **데이터 검증:** `dataset_validation.py`의 검증 항목을 확인하고 데이터 준비·평가 과정에 반영한다.
- **실험 기록 보완:** 각 실행의 코드 버전, 학습률, 시드, 데이터 분할, 사전학습 checkpoint 및 전체 설정을 함께 저장한다.

## 데이터(Datasets)

- **PPBR_AZ:** 현재 중간 결과의 downstream 회귀 데이터셋이다. 코드에서 사용하는 CSV 열은 SMILES 입력 `SMILES`, 정답 `Y`이다.
- **PubChem:** 사전학습용 분자 구조 데이터. [SDF 다운로드](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/)
- **ChEMBL:** 추가 분자 데이터 출처. [ChEMBL](https://www.ebi.ac.uk/chembl/)

실행에 필요한 원본 CSV/SDF와 사전학습 checkpoint는 별도로 준비해야 한다. 각 실행 파일의 `origin_raw_data`, 사전학습 데이터 경로 및 checkpoint 경로를 실제 파일 위치에 맞게 설정한다.

## 연구 환경 및 라이브러리(Environment)

보고된 연구 환경:

| 항목 | 환경 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| PyTorch | 2.14.1+cu132 |
| CUDA | 13.2 |

주요 라이브러리: `torch`, `torch_geometric`, `rdkit`, `pandas`, `matplotlib`, `pyarrow`, `umap-learn`, `setuptools`, `torchvision`.

현재 `requirements.txt`에는 `torch==2.13.0`이 기록되어 있어 위 연구 환경과 일치하지 않는다. 환경 재현 시 PyTorch·torchvision 및 CUDA 호환 버전을 확인하고 의존성 목록을 맞춰야 한다.

## 설치 및 실행(Getting Started)

### 1. 저장소 복제 및 가상환경 생성

```bash
git clone https://github.com/astatinegithub/Isonet.git
cd Isonet
python -m venv chem
```

Windows PowerShell에서는 `./chem/Scripts/Activate.ps1`, Linux/macOS에서는 `source chem/bin/activate`로 가상환경을 활성화한다.

### 2. 의존성 설치

위 환경 차이를 확인하여 필요한 버전을 정한 뒤 설치한다.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

CUDA 사용 시 GPU 환경에 맞는 PyTorch 설치 명령은 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 확인한다. `pip install -e .`은 로컬 `isonet` 패키지를 등록하며, 현재 `setup.py`는 의존성을 자동으로 설치하지 않는다.

### 3. 데이터·설정 준비 후 실행

저장소 루트에서 필요한 실험을 선택하여 실행한다.

```bash
# Stereo feature ON/OFF 비교
python -m isonet.experiment.ex01.main_01

# 자기지도 사전학습 유무 비교
python -m isonet.experiment.ex02.main_02

# Stereo Attention ON/OFF 비교
python -m isonet.experiment.ex05.main_05
```

- 각 실행 파일의 `__main__`에서 데이터 경로, `experiment_name`, 모델 설정, `train_seed`, `split_seed` 및 checkpoint 경로를 설정한다. 현재 스크립트는 한 번 실행할 때 한 학습 시드를 사용한다.
- 현재 downstream 실행 코드의 train/validation/test 분할 비율은 **70/20/10**, `split_seed`는 **25**이다. 학습 시드와 데이터 분할 시드는 별도 설정이다.
- 새 결과는 `isonet/experiment/exNN/output/{experiment_name}/`에 저장된다. `temp/`는 기존 결과 보관 경로이다.
- 처리된 데이터는 `isonet/dataset/processed_data/`에서 재사용된다. 전처리나 입력 feature를 변경할 때는 새로운 실험 이름·캐시 경로를 사용하여 이전 데이터를 잘못 재사용하지 않도록 한다.
- `ex03`의 기본 실행 설정은 checkpoint에서 사전학습을 재개한다. 처음부터 사전학습하려면 `checkpoint_path=None`으로 설정한다.
- `ex05`의 실험 이름에는 `lr1e4`가 포함되어 있지만 현재 호출부는 학습률을 전달하지 않아 `model_finetuning` 기본값인 **1e-3**을 사용한다. 폴더 이름만으로 과거 실행의 실제 학습률을 확정할 수 없다.
