# Isonet

## 도입(Introduction)

This project predicts ADMET properties from SMILES using a **Directed Message-Passing Neural Network (DMPNN)**.

본 프로젝트는 SMILES로부터 분자 그래프를 구성하고, DMPNN을 이용하여 ADMET 물성을 예측하는 연구이다. 입체화학(stereochemistry) feature, 자기지도 사전학습, Stereo Attention이 예측 성능에 미치는 영향을 비교한다.

현재 결과는 **PPBR_AZ(혈장 단백질 결합률) 회귀 예측**을 중심으로 정리하였다. 기존 DMPNN도 입체 태그를 활용할 수 있다는 점에서 출발하여, **CIP 정보를 어디에서 처리하는지와 입체중심 주변의 국소 Attention 보정이 추가 이득을 주는지**를 비교한다. 초기 Attention 비교 이후 CIP 전달 위치(`ex06`)와 encoder 고정 여부(`ex07`)를 추가로 검토하였다.

## 연구 가설(Research Hypothesis)

> **H1.** stereochemistry 정보를 atom/bond feature에 포함하면 stereo 정보를 포함하지 않은 모델보다 특히 stereochemical subset에서 ADMET 예측 성능이 향상될 것이다.

> **H2.** stereocenter와 그 주변 local representation을 별도의 attention mechanism으로 집약하면, 단순히 stereo feature만 제공하는 모델보다 stereochemical subset의 ADMET 예측 성능이 향상될 것이다.

추가로 마스킹 복원 사전학습의 효과를 비교한다. 아래 결과는 **전체 test set**에 대한 평가이며, stereochemical subset과 실제 이성질체 쌍의 물성 차이에 대한 검증은 후속 과제이다.

## 모델 구조(Model Architecture)

**SMILES → 분자 그래프 → DMPNN → Stereo Attention(선택) → Pooling → MLP → 물성 예측**

- **분자 그래프:** RDKit으로 원자·결합 feature를 구성한다. 입체화학 feature 사용 여부를 변경할 수 있다.
- **DMPNN:** 방향성 결합을 따라 메시지를 전달하여 원자별 표현을 생성한다.
- **Stereo Attention:** CIP 표지 R/S/r/s가 있는 입체중심을 query로, 직접 연결된 이웃 원자의 표현과 결합 feature를 key/value로 사용한다. 이웃 정보를 attention으로 집약하고 gate를 거쳐 입체중심의 표현에 더한다.
- **Pooling:** 원자별 표현을 Sum 또는 Mean pooling으로 집약하여 분자 표현을 만든다.
- **MLP:** 분자 표현으로부터 endpoint별 값을 예측한다.

Stereo Attention은 DMPNN 이후, pooling 이전에 적용된다. 현재 구현은 CIP 표지가 있는 원자 중심의 attention이며, E/Z 결합을 중심으로 별도의 attention을 수행하지는 않는다. `ex06`·`ex07`에서는 원자 CIP만 encoder에서 선택적으로 마스킹하고 **결합 stereo feature는 유지**한다. 일반 마스킹 사전학습은 encoder를 학습하며, Stereo Attention은 downstream 학습에서 학습한다.

## 진행 실험(Experiments)

| 실행 파일 | 비교 내용 | 결과 파일 대응 |
| --- | --- | --- |
| [ex01/main_01.py](isonet/experiment/ex01/main_01.py) | 원자 CIP·결합 stereo feature ON/OFF | `metrics_1.json`: ON / `metrics_2.json`: OFF |
| [ex02/main_02.py](isonet/experiment/ex02/main_02.py) | 마스킹 복원 사전학습 유무 | `1`: 사전학습 / `2`: 무작위 초기화 |
| [ex03/main_03.py](isonet/experiment/ex03/main_03.py) | 공용 encoder 사전학습·학습 재개 | `checkpoint1_1/ssl_checkpoint_*.pt` |
| [ex04/main_04.py](isonet/experiment/ex04/main_04.py) | Pooling 비교 | `1`: Sum / `2`: Mean |
| [ex05/main_05.py](isonet/experiment/ex05/main_05.py) | Stereo Attention ON/OFF | `1`: ON / `2`: OFF |
| [ex06/main_06.py](isonet/experiment/ex06/main_06.py) | Encoder CIP × Stereo Attention의 4조건 | `0`: 둘 다 OFF / `1`: CIP만 / `2`: Attention만 / `3`: 둘 다 ON |
| [ex07/main_07.py](isonet/experiment/ex07/main_07.py) | CIP를 받지 않는 사전학습 encoder 고정 + Attention ON/OFF | `0`: OFF / `1`: ON |

숫자는 `metrics_<번호>.json`에 대응한다. `ex08`은 2차 사전학습 비교를 준비하는 코드이며, 완료된 결과에 포함하지 않는다.

## 중간 결과(Interim Results)

아래 PPBR_AZ 결과는 저장소의 `temp/`에 보관된 `metrics_*.json`을 집계한 **학습 시드 3개에 대한 평균 ± 표본 표준편차**이다. 각 실행의 test sample 수는 163개이며 RMSE·MAE는 낮을수록 좋다. 같은 데이터 분할에서 학습 시드를 반복한 결과로, 다중 분할 재현성이나 통계적 유의성을 검증한 결과는 아니다.

이 표는 저장된 평가 지표의 요약이며, 현재 코드로 모든 과거 실험을 다시 학습하여 재현한 결과는 아니다. 각 비교군 내부의 ON/OFF를 비교하며, 서로 다른 실험군 사이의 차이를 특정 모듈 하나의 효과로 해석하지 않는다.

### 1. CIP 전달 위치와 Encoder 고정 (`ex06`·`ex07`)

결과: [`temp/06_cip_compare/`](temp/06_cip_compare/), [`temp/07_encoder_fix/`](temp/07_encoder_fix/) · 학습 시드: **20, 25, 42** · 분할 시드: **25** · Mean pooling

| 실험 | Encoder 학습 | Encoder CIP | Attention | RMSE ↓ | MAE ↓ |
| --- | --- | --- | --- | --- | --- |
| 06 | 미세조정 | OFF | OFF | 14.012 ± 0.345 | 9.873 ± 0.368 |
| 06 | 미세조정 | ON | OFF | **13.785 ± 0.506** | 9.910 ± 0.487 |
| 06 | 미세조정 | OFF | ON | 14.421 ± 0.483 | 9.965 ± 0.225 |
| 06 | 미세조정 | ON | ON | 14.175 ± 0.517 | **9.660 ± 0.402** |
| 07 | 고정 | OFF | OFF | 15.269 ± 0.014 | 10.852 ± 0.107 |
| 07 | 고정 | OFF | ON | 15.407 ± 0.157 | 10.714 ± 0.136 |

`ex06`에서는 encoder에만 CIP를 제공한 조건의 평균 RMSE가 가장 낮았지만, MAE는 두 경로를 모두 사용한 조건이 가장 낮았다. `ex07`에서도 Attention은 RMSE를 높이고 MAE를 낮춰 **지표 전반의 일관된 개선은 확인하지 못하였다**. Encoder 고정 조건의 오차는 대응하는 미세조정 조건보다 컸다. 결론은 이 분할과 설정의 범위에 한정한다.

본 연구는 입체정보의 입력 효과와 별도 국소 처리 모듈의 추가 효과를 구분하여 검토하였다. 정보 중복, 과적합, pooling에 따른 보정량 변화 등은 가능한 설명이며, 성능 저하의 원인으로 확정하지 않는다.

<details>
<summary>기존 Stereo feature·사전학습·Attention 비교 결과</summary>

### 2. Stereo feature

결과: [`temp/stereo_feature/`](temp/stereo_feature/) · 학습 시드: **25, 40, 42**

| Pooling | Stereo feature | RMSE ↓ | MAE ↓ |
| --- | --- | --- | --- |
| Sum | OFF | 14.301 ± 0.215 | 10.793 ± 0.164 |
| Sum | ON | 14.973 ± 0.414 | 11.592 ± 0.501 |
| Mean | OFF | 14.583 ± 0.106 | 9.986 ± 0.170 |
| Mean | ON | 14.029 ± 0.142 | 9.856 ± 0.062 |

Mean pooling에서는 feature 추가 시 평균 RMSE와 MAE가 감소했지만, Sum pooling에서는 증가하였다. 효과는 pooling 조건에 따라 달랐다.

### 3. 자기지도 사전학습

결과: [`temp/pretrain_rlt/`](temp/pretrain_rlt/)의 `*seed_re` 폴더 · 학습 시드: **20, 25, 42**

| 사전학습 | RMSE ↓ | MAE ↓ |
| --- | --- | --- |
| OFF | 13.602 ± 0.289 | 9.526 ± 0.377 |
| ON | 13.223 ± 0.200 | 9.329 ± 0.161 |

사전학습 적용 시 평균 RMSE는 약 2.8%, MAE는 약 2.1% 감소하였다. 이 비교만으로 입체화학 정보 학습의 효과를 분리할 수는 없다.

### 4. Stereo Attention

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

`*_add` 비교군에서는 평균 RMSE가 감소했으나 MAE는 개선되지 않았다. 초기 비교에서도 Attention의 일관된 추가 이득은 확인하지 못하였다.

</details>

## 후속 실험 계획(Future Work)

- **평가 확장:** Stereo subset·실제 이성질체 쌍의 물성 차이를 평가하고, 다른 endpoint와 외부 모델을 동일한 분할에서 비교한다.
- **모델 분석:** 학습·검증 손실, attention·gate 값과 보정량을 확인하고 pooling·모델 크기의 영향을 분리한다.
- **재현성 보완:** 데이터 검증 및 다중 분할 평가를 수행하고 코드 버전·전체 설정·원본 데이터·checkpoint를 함께 기록한다.

## 데이터(Datasets)

- **PPBR_AZ:** 현재 중간 결과의 downstream 회귀 데이터셋이다. 코드에서 사용하는 CSV 열은 SMILES 입력 `SMILES`, 정답 `Y`이다.
- **ExpansionRx_HLM:** 최신 `ex06`의 확장 실험 대상이다. `ExpansionRx_train.csv`·`ExpansionRx_test.csv`의 `SMILES`, `HLM CLint` 열을 사용하며, 현재 target 변환은 `log10(1 + y)`이다. 위 PPBR 표와 구분한다.
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

### 3. PPBR 결과 재현용 코드·입력 준비

**위 `ex06`·`ex07` PPBR 표의 재학습 기준은 `b84bfc9`이다.** 현재 `main`(`03dd07b`)은 `ex06`을 HLM·별도 test CSV로 변경하고 `ex07`을 250 epoch로 늘렸으므로 위 표의 조건과 다르다. PPBR 재현은 다음 버전에서 진행한다.

```bash
git fetch origin b84bfc94dd22c88c254f5f1174461efc109d1f0d
git checkout b84bfc94dd22c88c254f5f1174461efc109d1f0d
```

다음 파일을 준비한다. 경로는 저장소 루트 기준이며 원본 데이터와 공용 사전학습 checkpoint는 별도 제공이 필요하다.

| 파일 | 용도 |
| --- | --- |
| `isonet/dataset/raw/admet/PPBR_AZ.csv` | Downstream 입력: `SMILES`, `Y` |
| `isonet/experiment/ex03/output/pretrain_for_share_no_cip3/checkpoint1_1/ssl_checkpoint_250epoch.pt` | `ex06`·`ex07`이 공통으로 불러오는 encoder |
| `isonet/dataset/raw/smiles/Compound_001000001_001500000.sdf` | `ex03`에서 새 사전학습을 수행할 때 사용하는 SDF |

공용 checkpoint를 새로 만들려면 `ex03/main_03.py`의 `checkpoint_path=None`, `epochs=250`으로 설정하고 실행한다. 기본값은 기존 200-epoch checkpoint에서 50 epoch를 **재개**하는 설정이다. `encoder_use_cip=False`는 유지하며, 동일한 과거 결과를 대조하려면 당시 데이터·전처리 캐시와 **동일한 공용 checkpoint**가 필요하다.

```bash
# 공용 checkpoint를 새로 학습하는 경우에만 실행
python -m isonet.experiment.ex03.main_03
```

### 4. Downstream 실험 실행

저장소 루트에서 필요한 실험을 선택하여 실행한다. `ex06`·`ex07`은 각각 학습 시드 **20, 25, 42**를 모두 실행한다. `ex02`·`ex05`는 각 실행 파일에 지정된 별도 SDF·checkpoint 경로도 준비해야 한다.

```bash
# Stereo feature ON/OFF 비교
python -m isonet.experiment.ex01.main_01

# 자기지도 사전학습 유무 비교
python -m isonet.experiment.ex02.main_02

# Stereo Attention ON/OFF 비교
python -m isonet.experiment.ex05.main_05

# CIP 전달 위치 4조건 비교
python -m isonet.experiment.ex06.main_06

# 사전학습 encoder 고정 + Attention ON/OFF 비교
python -m isonet.experiment.ex07.main_07
```

- **PPBR 06·07 공통 조건:** Random split **70/20/10**, `split_seed=25`, Mean pooling, batch size **64**, AdamW, 학습률 **1e-3**, downstream **100 epoch**이다. 공용 encoder의 사전학습 250 epoch와 구분한다. 학습 시드와 분할 시드는 별도 설정이다.
- **설정 변경:** 각 실행 파일의 `main()`·`__main__`에서 데이터·checkpoint 경로, `experiment_name`, 모델 설정 및 시드를 지정한다. 초기 실험 표를 대조할 때도 pooling·사전학습 경로·시드를 해당 비교군에 맞춘다. `ex01`·`ex02`·`ex05` 기본 실행은 한 학습 시드이다.
- **캐시:** `isonet/dataset/processed_data/exNN_{experiment_name}/`를 재사용한다. 원본 데이터·전처리·분할 조건을 바꿀 때는 새 실험 이름을 사용한다.
- **평가·출력:** 회귀 모델은 validation MSE가 가장 낮은 checkpoint를 선택한다. `isonet/experiment/exNN/output/{experiment_name}/`에 `metrics_*.json`, `predictions*.pt`, loss curve와 checkpoint가 저장된다. `temp/`는 기존 결과 보관 경로이며 결과 표는 JSON의 RMSE·MAE를 시드별로 집계한다.
- **최신 코드에서 checkpoint 사용 시:** `03dd07b`의 `model_finetuning()`은 `pretrain_component`를 순회하므로 각 cfg에 `"pretrain_component": ["encoder"]`를 지정해야 한다. `ex05` 이름의 `lr1e4`도 실제 학습률을 보장하지 않으므로 호출값을 확인한다.
