from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig,GRPOTrainer
import os
import wandb
from datasets import load_dataset
from datasets import Dataset
from environment import reward_parseables,reward_matchings

os.environ["WANDB_PROJECT"] = "bbox_rl" 

model_name = "Qwen/Qwen3-VL-2B-Instruct"
dataset_id = 'UlrickBL/elevation-dataset-synthetic-v2'

SYSTEM_PROMPT = (
    ""
)

category = "window"


def preprocess_fn(example):
    img = example["image"]
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("RGB")
    img = smart_resize(img, 640)

    gt = example["ground_truth"]  # bounding box list

    prompt = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f'locate every instance that belongs to the following categories: {category}. '
                        f'For each window, report bbox coordinates, in JSON format like this: '
                        f'{{"bbox_2d": [x1, y1, x2, y2], "label": "{category}"}}'
            },
            {"type": "image"}
        ]
    }]
    return {
        "prompt": prompt,
        "answer": json.dumps(gt),
        "images": [img],
        "info": {"gt": gt}
    }

if __name__ == "__main__":

    processor = AutoProcessor.from_pretrained(model_name)

    train_dataset = load_dataset(dataset_id, split='train')
    train_dataset = train_dataset.map(preprocess_fn)
    train_dataset = train_dataset.shuffle(seed=42)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )


    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["qkv","gate_proj","up_proj"],
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    training_args = GRPOConfig(
        output_dir="Qwen3VL-2B-BBOX-Window",
        learning_rate=1e-4,
        temperature = 1.0,
        lr_scheduler_type = "cosine",
        warmup_steps = 30,
        beta = 0.02,
        save_strategy = "steps",
        remove_unused_columns=False,
        bf16=True,
        num_train_epochs=2,
        per_device_train_batch_size=3,
        max_completion_length=2024,
        num_generations=15,
        max_prompt_length=2048,
        gradient_accumulation_steps =10,
        report_to=["wandb"],
        logging_steps=1,
        save_steps=10,
        reward_weights = [0.2,0.8],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[reward_parseables,reward_matchings],
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()