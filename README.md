# Python VIBE-Automated Project

## Requirements
This is my experiment eviroument, pytorch0.4 should also be fine
- python >= 3.7
- matplotlib
- scikit-learn
- scipy
- pandas
- numpy
- opencv
- tqdm
- deap
- joblib
- etc

## Futrue Work
- [ ] 현재 레이블 들어오면 수동으로 되는데 이부분 수정해야됨(test.py)
- [ ] 2개 이상의 클래스가 학습될 경우, treshold_percent의 값보다 작을경우 3개 더뽑아서 PCA 재학습 되는데 이부분 수정(json입력도 해야됨)

## Usage
- config.py에서 train/test json 설정을 해줌
- train.py 로 실행(train의 경우)
- test.py 로 실행(test의 경우)

# Cite
### Citation
```
  @misc{seonwoolee,
    author = {Seon-Woo Lee, Ki-Shul Shin, Su-Woong Hong, In-Seo Song},
    title = {Python VIBE-Automated Project},
    year = {2020},
    howpublished = {\url{https://github.com/kichoul64/Mat2Py},
    note = {commit xxxxxxx}
  }
```
