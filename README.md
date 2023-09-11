# VIsualAILabCloneRepo


### Baseline : https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection

## 실험

1. 50 epoch 이후로 ema update
- 실험 근거 : student가 좀 더 수렴한 후에 ema update하는 것이 더 효과적인지 증명하기 위함.
- 기대 효과 : 만약 시험 근거가 참이라면, ema update의 방향성을 잡을 수 있음.

2. 10 epoch 이전에는 student와 teacher 모두 weak aug, 이후로는 student에 strong aug를 적용하되 20 epoch 이후부터 ema update
- 실험 근거 : 모델도 사람처럼 쉬운 문제부터 학습해가는게 맞는지 증명하기 위함
- 기대 효과 : 만약 실험 근거가 참이라면, 모델도 사람처럼 쉬운 문제부터 학습하는게 수렴에 도움이 됨을 증명할 수 있음.

3. No EMA & only Weak with 1:1 Sample
- 실험 근거 : 모델에 정말 EMA업데이트가 필요한가?를 확인하기 위한 비교군
- 기대 효과 : 모델에 EMA가 필요하다면 EMA를 한 모델보다 성능이 좋지 안게 나올 것임.

4. 2의 내용을 그대로 with 1:1 Sample
- 실험 근거 : 1:1 Sample의 효과를 확인하는 것이 필요함
- 기대 효과 : 1:1 Sample의 효과를 확인 가능

5. 2의 내용을 전부 2배 with 1:1 Sample
- 실험 근거 : 2보다 더 오래 학습하는 것이 좋은가 테스트
- 기대 효과 : 2보다 더 오래 학습하는 것이 좋은가 테스트

6. 2의 내용을 전부 1/2배 with 1:1 Sample
- 실험 근거 : 2보다 더 짧게 학습하는 것이 좋은가 테스트
- 기대 효과 : 2보다 더 짧게 학습하는 것이 좋은가 테스트