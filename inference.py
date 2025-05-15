import gc
import time
import torch
from transformers import BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from datasets import load_dataset

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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        quantization_config=bnb_config,
        use_cache=True
        )

else:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        use_cache=True
        )
    
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

print(f"Before adapter parameters: {model.num_parameters()}")
model.load_adapter("./output")
print(f"After adapter parameters: {model.num_parameters()}")

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

sample_data = test_dataset[0]
generated_text, actual_answer = text_generator(sample_data, processor, model, device, MAX_SEQ_LEN=128)
print(f"Generated Answer: {generated_text}")
print(f"Actual Answer: {actual_answer}")