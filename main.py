import os
import torch
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from transformers import BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv()

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
def collate_fn(examples, processor):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    image_inputs = [example[1]["content"][0]["image"] for example in examples]

    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = batch["input_ids"]

    return batch



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
    EPOCHS = 1
    BATCH_SIZE = 1
    GRADIENT_CHECKPOINTING = True,  # Tradeoff between memory efficiency and computation time.
    USE_REENTRANT = False,
    OPTIM = "paged_adamw_32bit"
    LEARNING_RATE = 2e-5
    LOGGING_STEPS = 50
    EVAL_STEPS = 50
    SAVE_STEPS = 50
    EVAL_STRATEGY = "steps"
    SAVE_STRATEGY = "steps"
    METRIC_FOR_BEST_MODEL="eval_loss"
    LOAD_BEST_MODEL_AT_END=True
    MAX_GRAD_NORM = 1
    WARMUP_STEPS = 0
    DATASET_KWARGS={"skip_prepare_dataset": True} # We have to put for VLMs
    REMOVE_UNUSED_COLUMNS = False # VLM thing
    MAX_SEQ_LEN=128
    NUM_STEPS = (283 // BATCH_SIZE) * EPOCHS
    print(f"NUM_STEPS: {NUM_STEPS}")

    train_dataset, eval_dataset, test_dataset = load_dataset("HuggingFaceM4/ChartQA", 
                                                         split=["train[:1%]", "val[:1%]", "test[:1%]"])
    """
    Dataset({
        features: ['image', 'query', 'label', 'human_or_machine'],
        num_rows: 283
    })
    """
    print(len(train_dataset))
    print("-"*30)
    print(train_dataset)
    print("-"*30)
    print(train_dataset[0])
    print("-"*30)

    train_dataset = [format_data(sample) for sample in train_dataset]
    eval_dataset = [format_data(sample) for sample in eval_dataset]
    test_dataset = [format_data(sample) for sample in test_dataset]

    print(len(train_dataset))
    print("-"*30)
    print(train_dataset[0])
    print("-"*30)
    print(len(test_dataset))
    print("-"*30)
    print(test_dataset[0])

    sample_data = test_dataset[0]
    sample_question = test_dataset[0][1]["content"][1]["text"]
    sample_answer = test_dataset[0][2]["content"][0]["text"]
    sample_image = test_dataset[0][1]["content"][0]["image"]

    print(sample_question)
    print(sample_answer)
    sample_image

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
            use_cache=False
            )

    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, 
            use_cache=False
            )
        
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"

    if device == "cuda":
        model.to(device)

    model.config.use_cache = False
    

    generated_text, actual_answer = text_generator(sample_data, processor, model, device, MAX_SEQ_LEN)
    print(f"Generated Answer: {generated_text}")
    print(f"Actual Answer: {actual_answer}")

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    print(f"Before adapter parameters: {model.num_parameters()}")
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters() # After LoRA trainable parameters increases. Since we add adapter.
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
        remove_unused_columns = REMOVE_UNUSED_COLUMNS,
        optim=OPTIM,
    )
    collate_sample = [train_dataset[0], train_dataset[1]] # for batch size 2.

    collated_data = collate_fn(collate_sample, processor)
    print(collated_data.keys())  # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'labels'])    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda examples: collate_fn(examples, processor),
        peft_config=peft_config,
        processing_class=processor.tokenizer,
    )
    print("-"*30)
    print("Initial Evaluation")
    metric = trainer.evaluate()
    print(metric)
    print("-"*30)

    print("Training")
    trainer.train()
    print("-"*30)

    trainer.save_model(training_args.output_dir)

    
if __name__ == "__main__":
    main()
