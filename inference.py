import gc
import time
import torch
from transformers import BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt

def text_generator(sample_data, processor, model, device, MAX_SEQ_LEN):
    text = processor.apply_chat_template(
        sample_data[0:2], tokenize=False, add_generation_prompt=True
    )

    print(f"Prompt: {text}")
    print("-"*30)

    image_inputs = sample_data[1]["content"][0]["image"]

    inputs = processor(
        text=[text],
        images = image_inputs,
        return_tensors="pt"
    )
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=MAX_SEQ_LEN)

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )
    del inputs
    actual_answer = sample_data[2]["content"][0]["text"]
    return output_text[0], actual_answer

def format_data(sample):
    system_message = """You are a highly advanced Vision Language Model (VLM), specialized in analyzing, describing, and interpreting visual data. 
    Your task is to process and extract meaningful insights from images, videos, and visual patterns, 
    leveraging multimodal understanding to provide accurate and contextually relevant information."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample["query"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"][0]}],
        },
    ]

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl
def clear_memory():
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "trainer" in globals():
        del globals()["trainer"]
    if "peft_model" in globals():
        del globals()["peft_model"]
    if "bnb_config" in globals():
        del globals()["bnb_config"]
    time.sleep(2)

    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


clear_memory()

if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    pretrained_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        quantization_config=bnb_config,
        use_cache=True
        )
    fine_tuned_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        quantization_config=bnb_config,
        use_cache=True
        )
else:
    pretrained_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        use_cache=True
        )
    fine_tuned_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        use_cache=True
        )
    
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

print(f"Before adapter parameters: {fine_tuned_model.num_parameters()}")
fine_tuned_model.load_adapter("./output")
print(f"After adapter parameters: {fine_tuned_model.num_parameters()}")

_, _, test_dataset = load_dataset("HuggingFaceM4/ChartQA", 
                                                         split=["train[:1%]", "val[:1%]", "test[:1%]"])
"""
Dataset({
    features: ['image', 'query', 'label', 'human_or_machine'],
    num_rows: 283
})
"""
print(len(test_dataset))
print("-"*30)
print(test_dataset)
print("-"*30)
print(test_dataset[0])
print("-"*30)

test_dataset = [format_data(sample) for sample in test_dataset]

# 결과를 저장할 리스트
pretrained_results = []
fine_tuned_results = []
actual_answers = []
images = []

# 여러 샘플에 대해 결과 수집 (예: 처음 3개 샘플)
for i in range(min(3, len(test_dataset))):
    sample_data = test_dataset[i]
    image = sample_data[1]["content"][0]["image"]
    
    # Pretrained 모델 결과
    pretrained_text, actual_answer = text_generator(sample_data, processor, pretrained_model, device, MAX_SEQ_LEN=128)
    pretrained_results.append(pretrained_text)
    
    # Fine-tuned 모델 결과
    fine_tuned_text, _ = text_generator(sample_data, processor, fine_tuned_model, device, MAX_SEQ_LEN=128)
    fine_tuned_results.append(fine_tuned_text)
    
    actual_answers.append(actual_answer)
    images.append(image)

# 결과 시각화
for i in range(len(pretrained_results)):
    plt.figure(figsize=(15, 10))
    
    # 이미지 표시
    plt.subplot(2, 1, 1)
    plt.imshow(images[i])
    plt.axis('off')
    plt.title(f"Sample {i+1} - Input Image")
    
    # 결과 텍스트 표시
    plt.subplot(2, 1, 2)
    plt.axis('off')
    result_text = f"Pretrained Model:\n{pretrained_results[i]}\n\n" \
                 f"Fine-tuned Model:\n{fine_tuned_results[i]}\n\n" \
                 f"Actual Answer:\n{actual_answers[i]}"
    plt.text(0.1, 0.5, result_text, fontsize=10, va='center')
    
    plt.tight_layout()
    plt.show()

# 메모리 정리
clear_memory()