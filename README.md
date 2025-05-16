# Qwen 2.5 VL 모델 학습 프로젝트

이 프로젝트는 Qwen 2.5 VL(Vision-Language) 모델을 ChartQA 데이터셋을 사용하여 fine-tuning하는 코드를 제공합니다.

## 주요 함수 및 파라미터 설명

### 데이터 포맷 함수

```python
def format_data(sample):
```

- **기능**: 데이터셋의 샘플을 모델 입력 형식으로 변환
- **파라미터**:
  - `sample`: ChartQA 데이터셋의 샘플 (이미지, 쿼리, 레이블 포함)
- **반환값**: 시스템 메시지, 사용자 입력(이미지+텍스트), 어시스턴트 응답을 포함하는 리스트
- **설명**: 
  - 시스템 메시지로 VLM의 역할을 정의
  - 사용자 입력으로 이미지와 쿼리를 포함
  - 어시스턴트 응답으로 정답 레이블을 포함

### 텍스트 생성 함수

```python
def text_generator(sample_data, processor, model, device, MAX_SEQ_LEN):
```

- **기능**: 주어진 샘플에 대해 모델의 텍스트 응답을 생성
- **파라미터**:
  - `sample_data`: 포맷된 데이터 샘플
  - `processor`: 텍스트와 이미지를 처리하는 프로세서
  - `model`: Qwen 2.5 VL 모델
  - `device`: 연산 장치 (CPU/GPU)
  - `MAX_SEQ_LEN`: 생성할 최대 토큰 수
- **반환값**: 생성된 텍스트와 실제 정답
- **설명**:
  - 프롬프트를 생성하고 출력
  - 이미지와 텍스트를 모델 입력으로 변환
  - 모델을 통해 텍스트 생성
  - 생성된 텍스트와 실제 정답 반환

### 데이터 배치 처리 함수

```python
def collate_fn(examples, processor):
```

- **기능**: 여러 샘플을 하나의 배치로 처리
- **파라미터**:
  - `examples`: 포맷된 데이터 샘플 리스트
  - `processor`: 텍스트와 이미지를 처리하는 프로세서
- **반환값**: 배치 처리된 데이터 (input_ids, attention_mask, pixel_values, labels)
- **설명**:
  - 각 샘플에 대해 채팅 템플릿 적용
  - 이미지 입력 추출
  - 텍스트와 이미지를 배치로 처리
  - 레이블 생성 및 패딩 토큰 처리

### 메인 함수

```python
def main():
```

- **기능**: 모델 학습 파이프라인 실행
- **주요 파라미터**:
  - `MODEL_ID`: 사용할 모델 ID ("Qwen/Qwen2.5-VL-3B-Instruct")
  - `EPOCHS`: 학습 에폭 수 (1)
  - `BATCH_SIZE`: 배치 크기 (1)
  - `LEARNING_RATE`: 학습률 (2e-5)
  - `MAX_SEQ_LEN`: 최대 시퀀스 길이 (128)
- **설명**:
  1. 장치 설정 (CPU/GPU)
  2. ChartQA 데이터셋 로드 및 전처리
  3. 모델 및 프로세서 초기화
  4. 샘플 데이터로 모델 테스트
  5. LoRA 설정 구성
  6. 학습 설정 구성
  7. SFTTrainer를 사용한 모델 학습
  8. 모델 평가 및 저장

## LoRA 설정

```python
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
```

- **lora_alpha**: LoRA 스케일링 파라미터 (16)
- **lora_dropout**: LoRA 레이어의 드롭아웃 비율 (0.1)
- **r**: LoRA의 랭크 (8)
- **bias**: 바이어스 파라미터 학습 여부 ("none")
- **target_modules**: LoRA를 적용할 모듈 (["q_proj", "v_proj"])
- **task_type**: 태스크 유형 ("CAUSAL_LM")

## 학습 설정

```python
training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
    eval_steps=EVAL_STEPS,
    eval_strategy=EVAL_STRATEGY,
    save_strategy=SAVE_STRATEGY,
    save_steps=SAVE_STEPS,
    metric_for_best_model=METRIC_FOR_BEST_MODEL,
    load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
    max_grad_norm=MAX_GRAD_NORM,
    warmup_steps=WARMUP_STEPS,
    dataset_kwargs=DATASET_KWARGS,
    max_seq_length=MAX_SEQ_LEN,
    remove_unused_columns=REMOVE_UNUSED_COLUMNS,
    optim=OPTIM,
)
```

- **output_dir**: 모델 저장 경로 ("./output")
- **num_train_epochs**: 학습 에폭 수 (1)
- **per_device_train_batch_size**: 학습 배치 크기 (1)
- **per_device_eval_batch_size**: 평가 배치 크기 (1)
- **gradient_checkpointing**: 그래디언트 체크포인팅 사용 여부 (True)
- **learning_rate**: 학습률 (2e-5)
- **logging_steps**: 로깅 단계 (50)
- **eval_steps**: 평가 단계 (50)
- **save_steps**: 저장 단계 (50)
- **max_seq_length**: 최대 시퀀스 길이 (128)
- **optim**: 옵티마이저 ("paged_adamw_32bit")

## 사용 방법

1. 필요한 패키지 설치
2. ChartQA 데이터셋 준비
3. main.py 실행
4. 학습된 모델은 "./output" 디렉토리에 저장됨
5. inference.py를 사용하여 학습된 모델 평가

## 참고

- 이 코드는 Qwen 2.5 VL 모델을 ChartQA 데이터셋의 일부(1%)로 fine-tuning합니다.
- LoRA를 사용하여 효율적인 학습을 수행합니다.
- 학습된 모델은 차트 이미지에 대한 질문에 답변하는 능력이 향상됩니다.