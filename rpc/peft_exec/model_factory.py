from __future__ import annotations

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForImageClassification,
    AutoTokenizer,
    AutoImageProcessor,
)
from peft import LoraConfig, get_peft_model, TaskType

BACKBONE_REGISTRY = {
    "distilbert-base-uncased": {
        "target_modules": ["q_lin", "v_lin"],
        "task_type": TaskType.SEQ_CLS,
        "modules_to_save": ["pre_classifier", "classifier"],
    },
    "distilbert-base-uncased-finetuned-sst-2-english": {
        "target_modules": ["q_lin", "v_lin"],
        "task_type": TaskType.SEQ_CLS,
        "modules_to_save": ["pre_classifier", "classifier"],
    },
    "bert-base-uncased": {
        "target_modules": ["query", "value"],
        "task_type": TaskType.SEQ_CLS,
        "modules_to_save": ["classifier"],
    },
    "FacebookAI/roberta-base": {
        "target_modules": ["query", "value"],
        "task_type": TaskType.SEQ_CLS,
        "modules_to_save": ["classifier"],
    },
    "google/vit-base-patch16-224": {
        "target_modules": ["query", "value"],
        "task_type": None,
        "modules_to_save": ["classifier"],
    },
    "WinKawaks/vit-small-patch16-224": {
        "target_modules": ["query", "value"],
        "task_type": None,
        "modules_to_save": ["classifier"],
    },
}

DATASET_META = {
    "sst2": {"num_labels": 2, "task": "text_classification"},
    "agnews": {"num_labels": 4, "task": "text_classification"},
    "qnli": {"num_labels": 2, "task": "text_classification"},
    "mrpc": {"num_labels": 2, "task": "text_classification"},
    "mnli": {"num_labels": 3, "task": "text_classification"},
    "rte": {"num_labels": 2, "task": "text_classification"},
    "cola": {"num_labels": 2, "task": "text_classification"},
    "imdb": {"num_labels": 2, "task": "text_classification"},
    "cifar100": {"num_labels": 100, "task": "image_classification"},
    "tiny_imagenet": {"num_labels": 200, "task": "image_classification"},
}

def create_model(
    backbone: str,
    num_labels: int,
    task_type: str = "text_classification",
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    precision: str = "bf16",
):
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    if task_type == "image_classification":
        base_model = AutoModelForImageClassification.from_pretrained(
            backbone, num_labels=num_labels, ignore_mismatched_sizes=True,
            torch_dtype=dtype,
        )
        processor = AutoImageProcessor.from_pretrained(backbone)
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            backbone, num_labels=num_labels, torch_dtype=dtype,
        )
        processor = AutoTokenizer.from_pretrained(backbone)

    reg = BACKBONE_REGISTRY.get(backbone, {})
    modules = target_modules or reg.get("target_modules", ["query", "value"])
    peft_task = reg.get("task_type", TaskType.SEQ_CLS)
    modules_to_save = reg.get("modules_to_save", None)

    lora_kwargs = dict(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=modules,
        modules_to_save=modules_to_save,
        bias="none",
    )
    if peft_task is not None:
        lora_kwargs["task_type"] = peft_task

    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(base_model, lora_config)
    return model, processor
