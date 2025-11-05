# LoRA Fine-Tuning for Drug Interaction QA (RTX 3050 Ti Friendly)

This repository contains two complete LoRA training pipelines for drug-interaction question answering, both optimized for a 4 GB RTX 3050 Ti using 4-bit QLoRA:

- **Model A – TinyLlama/TinyLlama-1.1B-Chat-v1.0** (`LoRA.ipynb`)
- **Model B – Qwen/Qwen2-0.5B-Instruct** (`Qwen.ipynb`)

Both notebooks load the cleaned TwoSides/OffSides datasets, fine-tune a base model with LoRA adapters, and ship with evaluation + confusion-matrix tooling.

---

## Project Structure

```
LoRA_FT/
├─ LoRA.ipynb                         # TinyLlama LoRA pipeline
├─ Qwen.ipynb                         # Qwen LoRA pipeline
├─ outputs/
│  ├─ drug_lora_model/                # TinyLlama adapters + tokenizer + metrics
│  └─ qwen2-0.5b-instruct_<timestamp>/ # Qwen adapters + tokenizer + metrics
└─ README.md                          # This document
```

> The CSV datasets (`TwoSidesData.csv`, `OffSidesData.csv`) are large and should live alongside the notebooks locally, but are intentionally excluded from Git.

---

## Environment Setup (Windows + PowerShell)

```powershell
py -3.12 -m venv venv
venv\Scripts\activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft datasets accelerate bitsandbytes pandas numpy scikit-learn matplotlib seaborn
```

You can also let each notebook install missing packages inline (it checks on import).

---

## Dataset Expectations

- `TwoSidesData.csv`: `drug_1_concept_name`, `drug_2_concept_name`, `condition_concept_name`, `PRR`
- `OffSidesData.csv`: `drug_concept_name`, `condition_concept_name`, `PRR`

Both notebooks randomly sample from these tables each run, so they work even if the source CSVs are very large.

---

## Model A – TinyLlama (LoRA.ipynb)

**Purpose:** Fast baseline LoRA fine-tuning (~15–60 minutes). Useful for quick experiments or small-batch training.

### Notebook Flow

1. CUDA/GPU check
2. Load datasets and sample (default 1K–5K rows)
3. Build instruction-style Q&A prompts
4. Load TinyLlama 1.1B Chat in 4-bit NF4
5. Apply LoRA (r=8, α=16)
6. Tokenize + collate
7. Train (default 1–3 epochs, grad-acc 8)
8. Save adapters/tokenizer to `outputs/drug_lora_model/`
9. Quick QA test (Cell 10)
10. Accuracy + partial match evaluation (Cell 11)
11. Confusion matrix and classification report (Cell 12)

### Output Directory

```
LoRA_FT/outputs/drug_lora_model/
├─ adapter_model.safetensors
├─ adapter_config.json
├─ tokenizer.json / tokenizer.model / tokenizer_config.json / special_tokens_map.json
├─ confusion_matrix.png
└─ checkpoints/... (latest training checkpoints)
```

### Sample TinyLlama Responses

- **Aspirin + Warfarin →** “Bone marrow failure”
- **Metformin →** “Blood bicarbonate decreased”
- **Ibuprofen + Naproxen →** “Dysuria”

> Outputs may include extra commentary; post-process the `Answer:` section if you need only the condition.

### TinyLlama Inference Snippet

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_dir = r"C:\Users\Muham\OneDrive\Desktop\LoRA_FT\outputs\drug_lora_model"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", load_in_4bit=True, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()

def answer(question: str):
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, temperature=0.3, do_sample=True, top_p=0.9,
                                pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True).split("### Answer:")[-1].strip()

print(answer("What adverse event might occur when taking aspirin and warfarin together?"))
```

---

## Model B – Qwen 0.5B (Qwen.ipynb)

**Purpose:** Higher-quality LoRA fine-tuning with longer runs (configured ~6–7 hours). Uses larger training sets and more epochs for lower loss.

### Notebook Flow

1. Same dataset preparation as TinyLlama
2. Load Qwen/Qwen2-0.5B-Instruct in 4-bit NF4 with gradient checkpointing
3. Apply LoRA (r=8, α=16, expanded target modules)
4. Tokenize + collate
5. Train with scaled hyperparameters:
   - `TRAIN_SIZE = 20_000`
   - `EPOCHS = 6`
   - `GRAD_ACC = 16`
   - `LR = 1e-4`
   - `warmup_ratio = 0.1`
   - `logging_steps = 20`, `save_steps = 500`
6. Save adapters/tokenizer to `outputs/qwen2-0.5b-instruct_<timestamp>/`
7. Cell 10: quick sanity check
8. Cell 11: full accuracy evaluation on 400 held-out samples
9. Cell 12: confusion matrix + classification report

### Output Directory Example

```
LoRA_FT/outputs/qwen2-0.5b-instruct_20251030-180847/
├─ adapter_model.safetensors
├─ adapter_config.json
├─ tokenizer.json / tokenizer.model / tokenizer_config.json / special_tokens_map.json
├─ confusion_matrix.png
├─ merges.txt / vocab.json / added_tokens.json
└─ checkpoint-XXXX/ (multiple intermediate checkpoints)
```

### Training Expectations

- Fits within 4 GB VRAM thanks to 4-bit quantization + gradient checkpointing
- Run time ~6–7 hours with the hyperparameters above
- Loss keeps dropping across six epochs; monitor with `nvidia-smi -l 1`

### Sample Qwen Responses

- **Aspirin + Warfarin →** “Bone marrow failure” (+ richer contextual comments)
- **Metformin →** “Blood bicarbonate decreased” (+ extra clarifying sentences)
- **Ibuprofen + Naproxen →** “Dysuria”

> Qwen tends to add longer explanations. Parse the first sentence or the portion after “Answer:” if you need just one label.

### Qwen Inference Snippet

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base = "Qwen/Qwen2-0.5B-Instruct"
adapter_dir = r"C:\Users\Muham\OneDrive\Desktop\LoRA_FT\outputs\qwen2-0.5b-instruct_20251030-180847"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
model = AutoModelForCausalLM.from_pretrained(base, load_in_4bit=True, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()

print(answer("What adverse event might occur when taking metformin?"))
```

(Reuse the `answer` helper from the TinyLlama example.)

---

## Evaluation Outputs

Both notebooks share the same evaluation logic:

- **Accuracy metrics** (exact + partial match)
- **Confusion matrix** for top-10 conditions
- **classification_report** (precision/recall/F1)

Artifacts are written to each model’s output folder, e.g.

```
LoRA_FT/outputs/drug_lora_model/confusion_matrix.png
LoRA_FT/outputs/qwen2-0.5b-instruct_20251030-180847/confusion_matrix.png
```

> Exact match can be strict (synonyms count as mismatches). Partial-match accuracy is a better indicator for clinical terminology.

---

## Tips & Troubleshooting

- **Batch size / OOM:** keep `per_device_train_batch_size = 1`, adjust `GRAD_ACC` for effective batch size
- **Sequence length:** 256 works; drop to 128 if memory-constrained
- **Temperature:** lower (0.2–0.3) during evaluation for more deterministic answers
- **Checkpoint cleanup:** each run saves several checkpoints; delete older ones if disk space is tight
- **Tokenizer mismatch:** always load tokenizer from the saved adapter directory

---

## Reproducibility Notes

- QLoRA (4-bit quantization + LoRA adapters) keeps VRAM usage low
- Random sampling of the CSVs means runs differ slightly unless you fix seeds everywhere
- Update `BASE_DIR` if you move the project

---

## License & Intended Use

The fine-tuned models are for research/educational use only. Always verify drug-interaction outputs with clinical sources before acting on them.

---

## Acknowledgements

- TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Qwen/Qwen2-0.5B-Instruct
- Hugging Face `transformers`, `peft`, `datasets`, `accelerate`, `bitsandbytes`
- TwoSides / OffSides datasets
