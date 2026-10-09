# 도입(Introduction) 
This project is to create a model that makes predictions based on SMILE data using **Message-Passing Neural Network (DMPNN)**.

본 프로젝트는 smiles 데이터셋을 이용하여 admet을 예측해주는 프로젝트 입니다.


> !주의 코드에 주석이 달려있지 않아서 해석이 어려울 수 있음


# 연구 가설(Research Hypothesis)
> **H1.** stereochemistry 정보를 atom/bond feature에 포함하면 stereo 정보를 포함하지 않은 모델보다 특히 stereochemical subset에서 ADMET 예측 성능이 향상될 것이다.

> **H2.** stereocenter와 그 주변 local representation을 별도의 attention mechanism으로 집약하면, 단순히 stereo feature만 제공하는 모델보다 stereochemical subset의 ADMET 예측 성능이 향상될 것이다.


# 실험 계획(experiment plan)

###  파라미터 ablation study
`depth`, `hiddendim`같은 하이퍼파라미터를 조정할 예정


| 모델                                   | 구성                                              | 확인할 것                       |
| ------------------------------------ | -------------------------------------------------  | ------------------------------ |
| **Model A: Baseline**                | DMPNN + 일반 molecular feature                      | 기본 ADMET 성능                    |
| **Model B: Stereo Feature**          | A + atom chirality(R/S 관련 tag) + bond E/Z feature | stereo 정보를 단순 입력하는 것의 효과    |
| **Model C: Stereo Attention (Ours)** | B + stereocenter와 주변 1-hop 원자에 별도 attention  | stereo 관련 국소 정보를 강조하는 것의 추가 효과 |

#### 실험 1-1. Stereo feature 효과 비교  
`Model A vs Model B`를 비교한다. 이를 통해 R/S, E/Z 정보를 단순히 feature에 추가하는 것만으로 ADMET 예측이 개선되는지 확인한다.

#### 실험 1-2. Stereo Attention 효과 비교 — 핵심 실험  
`Model B vs Model C`를 비교한다. 이를 통해 stereo feature를 전체 embedding에 섞는 것보다 stereocenter 주변을 별도로 attention하는 것이 효과적인지 확인한다.

#### 실험 1-3. Stereo subset 평가  
전체 ADMET test set뿐 아니라 **R/S 또는 E/Z 정보가 존재하는 분자만 따로 추출한 subset**에서도 성능을 평가한다. (Stereo Attention의 효과가 전체 데이터에서 묻히는 것을 방지하기 위함)

#### 실험 1-4. 모델 분석  
stereocenter 주변 atom의 attention weight를 분석하고, 실제 데이터 안에 R/S 또는 E/Z pair가 충분히 있다면 두 이성질체의 **실제 ADMET 차이와 모델이 예측한 차이**도 추가로 비교한다.

### 실험 2
아래 모델들에 대한 성능평가를 진행하여 실제사용되는 모델들에 비해 우리 모델이 어떤 성능을 지녔는지 판별한다.
(0) our model
(1) Chemprop (DMPNN)
(2) GROVER(Transformer + GNN)
(3) MolFormer(SMILES Transformer)
(4) Mole-BERT(Graph BERT)



# 나중에 할일
- `dataset_validation.py`이해하기 -> 이해부족으로 활용이 어려움



# Library
- setuptools
- torch
- torch_geometric
- torchvision
- pandas
- rdkit
- matplotlib
- pyarrow # parquet읽기위해
- umap-learn

you can install these by pip:

### windows

```
python -m pip install -r requirements.txt
```


# dataset
- [PubChem api](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/)
- [chembl](https://www.ebi.ac.uk/chembl/)



# clone시 가이드
1. 가상환경제작

`python -m venv chem` 를 이용해 가상환경을 만든다.

2. 필요 라이브러리 install

`python -m pip install -r requirements.txt` # requirements.txt에 있는 