| 파일명            | 구성요소 설명                                               
| ---------------   | ------------------------------------------------------- 
| `README.md`       | 프로젝트 개요, 실행 방법, 구조 설명 등                               
| `evaluate.ipynb`  | `model_1.pth` vs `model_2.pth` 대국 후 승/패/무 집계           
| `game.ipynb`      | `Agent` 두 명이 `YinshEnv`에서 번갈아 플레이하는 `Game` 클래스 구현     
| `mapper.ipynb`    | `action_to_index()`, `index_to_action()`으로 MCTS ↔ NN 연결 
| `model.ipynb`     | `YinshNet`: Policy + Value head 출력하는 CNN                
| `selfplay.ipynb`  | MCTS 기반 에이전트가 대국하여 학습용 (state, policy, value) 생성    
| `test.ipynb`      | 학습된 모델의 입력 처리, 예측, 학습 가능 여부를 간단히 테스트  
| `train.ipynb`     | selfplay로 생성된 데이터 기반으로 모델 학습 (loss: CE + MSE)           
