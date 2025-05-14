---
layout: post
title: "💸 PoorOrpo: Finetuning LLMs on a budget"
---

<a href="https://colab.research.google.com/github/nishzsche/nishzsche.github.io/blob/gh-pages/PoorOrpo_Finetune_and_evaluate_an_LLM_on_a_budget.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 💸 PoorOrpo: Finetuning LLMs on a budget

This notebook is an condensed end-to-end LLM finetuning guide. It finetunes, evaluates, and infers an LLM. All from this little notebook. 🐁



```python
from google.colab import userdata

# @title ## ℹ 1. Setup your finetuning project

# @markdown Add your personal project information Colab secrets to deploy models, monitoring, and results.

# @markdown ---

# @markdown ### 🧑 Project Configuration

## @markdown The huggingface hub repo id to push the final quantized model:

TUNED_MODEL_ID = "burtenshaw/Qwen1.5-0.5B-dpo-mix-7k" # @param {type:"string"}
TUNED_MODEL_NAME = TUNED_MODEL_ID.split("/")[1]
## @markdown The weights & biases project to log results to during training:
WANDB_PROJECT="skintstack-orpo" # @param {type:"string"}

# @markdown ---
# @markdown ### 🔐 Credentials

HF_USERNAME="burtenshaw" # @param {type:"string"}
HF_TOKEN="HF_TOKEN" # @param {type:"string"}
WANDB_TOKEN= "WANDB_TOKEN" # @param {type:"string"}
# RUNPOD_TOKEN = "RUNPOD_TOKEN" # @param {type:"string"}
# GITHUB_TOKEN = "GITHUB_TOKEN" # @param {type:"string"}

HF_TOKEN = userdata.get('HF_TOKEN')
WANDB_TOKEN = userdata.get('WANDB_TOKEN')
# RUNPOD_TOKEN = userdata.get("RUNPOD_TOKEN")


# @markdown ---
```


```python
# @title ## 🚆 2. Finetune a base model with ORPO

# @markdown We will train the model using [ORPO](https://huggingface.co/papers/2403.07691), because it outperforms `SFT, SFT+DPO` on `PHI-2, Llama 2, and Mistral`.

# @markdown ---

# @markdown ### 🤗 Training Parameters

config = """compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false"""

with open("orpo/train.yml", "w") as f:
    f.write(config)

BASE_MODEL_ID = "Qwen/Qwen1.5-0.5B" # @param {type:"string"}
DATASET = "argilla/dpo-mix-7k" # @param {type:"string"}
EPOCH = 1 # @param {type:"integer"}
LEARNING_RATE = 5e-6 # @param {type:"number"}
BATCH_SIZE = 2 # @param [1,2,4,8,16]
TRUST_REMOTE_CODE = True

# @markdown ---
!pip install -qqq torch>=1.10 datasets accelerate wandb transformers bitsandbytes sentencepiece --progress-bar off
!wandb login $WANDB_TOKEN
!wandb init -p $WANDB_PROJECT
!git clone https://github.com/xfactlab/orpo.git

!cd orpo && accelerate launch --config_file train.yml main.py \
    --lr {LEARNING_RATE} \
    --warmup_steps 100 \
    --model_name {BASE_MODEL_ID} \
    --data_name {DATASET} \
    --num_train_epochs {EPOCH} \
    --prompt_max_length 128 \
    --response_max_length 2048 \
    --per_device_train_batch_size {BATCH_SIZE} \
    --per_device_eval_batch_size {BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --num_proc 1

OUTPUT = "checkpoints/"+ DATASET.split("/")[-1]
!echo "++++++ Publishing to the Hub ++++++"
!huggingface-cli login --token {HF_TOKEN}
!huggingface-cli upload {TUNED_MODEL_NAME} orpo/{OUTPUT}

```


```python
# @title # 🧐 3. Evaluate the finetuned model with (with [lm-evaluation-harness by EleutherAI](https://github.com/EleutherAI/lm-evaluation-harness))

!git clone https://github.com/EleutherAI/lm-evaluation-harness
!cd lm-evaluation-harness && pip install -qqq -e .[ifeval] --progress-bar off
!pip install accelerate
!pip install -q huggingface_hub

import json
from huggingface_hub import create_repo, HfApi, ModelCard, EvalResult, ModelCardData

# @markdown Select the benchmark you want to use.
benchmark="ifeval" #@param ["eq-bench", "ifeval", "hellaswag"]
publish_to_hub = True #@param
!echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [1/6] =================="
!accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained={TUNED_MODEL_ID},dtype=auto,trust_remote_code=true \
    --tasks {benchmark} \
    --num_fewshot 0 \
    --batch_size auto:4 \
    --output_path ./{TUNED_MODEL_ID}-{benchmark}.json \
    --verbosity "critical"

benchmark = "eq-bench"

with open(f"{TUNED_MODEL_ID}-{benchmark}.json") as f:
    results = json.load(f)

print(results)
if publish_to_hub:
    eval_results = []
    for benchmark, config in results["configs"].items():
        dataset_name = config["dataset_path"]
        for metric in config["metric_list"]:
            metric_name = metric.pop("metric")
            score = results["results"][benchmark][f"{metric_name},none"]
            eval_result = EvalResult(
                task_name=benchmark,
                task_type="question-answering",
                dataset_name=dataset_name,
                dataset_type="benchmark",
                metric_type="accuracy",
                metric_value=score,
                metric_name=metric_name,
                # metric_args=metric
            )
            eval_results.append(eval_result)



    # Create model card
    card_data = ModelCardData(
        language='en',
        license='mit',
        model_name=TUNED_MODEL_NAME,
        eval_results = eval_results
    )

    card = ModelCard.from_template(card_data)
    card.save(f'{TUNED_MODEL_NAME}/README.md')
    card.push_to_hub(TUNED_MODEL_ID, token=HF_TOKEN)
```

# 💡 4. Use the model with!

Use a HF serverless endpoint to infer with your new model! You could also deploy your model to ChatUI space from [here](https://huggingface.co/new-space?template=huggingchat/chat-ui-template).


```python
import requests

API_URL = f"https://api-inference.huggingface.co/models/{TUNED_MODEL_ID}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

output = query({
	"inputs": "Can you please let us know more details about your ",
})

print(output[0]["generated_text"])
```

    Can you please let us know more details about your 2019-2020 schedule? We are looking forward to hearing from you.
    We are currently in the process of updating our schedule. We will be posting updates as soon as we have them.
