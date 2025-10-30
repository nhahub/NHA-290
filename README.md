# LoRA Fine-Tuning for Drug Interaction QA (RTX 3050 Ti Friendly)

This repository contains a complete, reproducible pipeline to fine-tune a language model with LoRA (Low-Rank Adaptation) on large-scale drug interaction datasets and evaluate its performance. The project is optimized to run on a modest GPU (RTX 3050 Ti 4GB VRAM) using 4-bit quantization and memory-efficient training.

---

## Project Structure

```
LoRA_FT/
├─ LoRA.ipynb                         # Main end‑to‑end notebook (recommended)
├─ TwoSidesData.csv                   # Drug–drug → condition (PRR) dataset
├─ OffSidesData.csv                   # Drug → condition (PRR) dataset
├─ outputs/
│  └─ drug_lora_model/               # Fine‑tuned adapter + tokenizer
│     ├─ adapter_model.safetensors   # LoRA adapter weights
│     ├─ adapter_config.json         # LoRA adapter config
│     ├─ tokenizer.json              # Tokenizer files
│     ├─ tokenizer.model
│     ├─ tokenizer_config.json
│     ├─ special_tokens_map.json
│     └─ confusion_matrix.png        # Saved during evaluation (Cell 12)
└─ README.md                         # This document
```

---

## Environment Setup (Windows, PowerShell)

You only need a recent Python (3.12 recommended) and an NVIDIA GPU with drivers + CUDA runtime installed (you already have them).

1) Create and activate a venv (optional but recommended):

```powershell
py -3.12 -m venv venv
venv\Scripts\activate
```

2) Install core Python packages (the notebook will also install what it needs):

```powershell
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft datasets accelerate bitsandbytes pandas numpy scikit-learn matplotlib seaborn
```

> Note: If you prefer, simply run the notebook – it installs missing packages inside the cells.

---

## Dataset Expectations

- `TwoSidesData.csv` columns: `drug_1_concept_name`, `drug_2_concept_name`, `condition_concept_name`, `PRR`
- `OffSidesData.csv` columns: `drug_concept_name`, `condition_concept_name`, `PRR`

Large files are supported; the notebook samples manageable amounts for quick runs and can be scaled up.

---

## End‑to‑End Flow (LoRA.ipynb)

Open `LoRA.ipynb` and run the cells top‑to‑bottom. The notebook is organized as:

- Cell 1: CUDA and GPU check
- Cell 2: Load datasets (from the project root)
- Cell 3: Prepare instruction‑style training samples (configurable size)
- Cell 4: Load base model with 4‑bit quantization (TinyLlama 1.1B Chat)
- Cell 5: Configure and apply LoRA adapters
- Cell 6: Tokenize dataset
- Cell 7: Configure `TrainingArguments` (epochs, LR, batch size, etc.)
- Cell 8: Train (15–60 minutes depending on settings)
- Cell 9: Save LoRA adapter + tokenizer to `outputs/drug_lora_model/`
- Cell 10: Quick interactive testing
- Cell 11: Accuracy evaluation (exact/partial match)
- Cell 12: Confusion matrix for top conditions (saved as PNG)

### Recommended Training Settings

- Base Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (fits in 4GB)
- Quantization: 4‑bit NF4 via `bitsandbytes`
- LoRA: `r=8`, `alpha=16`, targets `q_proj,k_proj,v_proj,o_proj`
- Training size: 1,000 → 5,000 examples (increase as VRAM/time allow)
- Epochs: 1 → 3 (more epochs → lower loss)
- Effective batch size: gradient accumulation (e.g., 1×8)

---

## Using the Fine‑Tuned Output

The trained artifacts are saved to:

```
C:\Users\Muham\OneDrive\Desktop\LoRA_FT\outputs\drug_lora_model
```

Key files:
- `adapter_model.safetensors` + `adapter_config.json`: LoRA adapter
- `tokenizer.*`: Tokenizer used during training
- `confusion_matrix.png`: Generated in evaluation

### Load for Inference (Python)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_dir = r"C:\\Users\\Muham\\OneDrive\\Desktop\\LoRA_FT\\outputs\\drug_lora_model"

# Load tokenizer from adapter_dir (ensures consistency)
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

# Load base model in 4‑bit
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)

# Attach LoRA adapter
model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()

# Ask a question
prompt = "### Question:\nWhat adverse event might occur when taking aspirin and warfarin together?\n\n### Answer:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = response.split("### Answer:")[-1].strip()
print(answer)
```

---

## Evaluation (Accuracy + Confusion Matrix)

The notebook includes two evaluation cells:

- **Cell 11 – Accuracy**
  - Creates a held‑out test set (not used for training)
  - Computes Exact‑Match Accuracy and Partial‑Match Accuracy
  - Prints sample predictions with ✓/✗ flags

- **Cell 12 – Confusion Matrix**
  - Focuses on top‑10 most frequent conditions in the test set
  - Saves a heatmap as `outputs/drug_lora_model/confusion_matrix.png`
  - Prints a `classification_report` summarizing precision/recall/F1

> Note: Because this is **generative** QA (not strict classification), exact match can be harsh. Partial‑match accuracy is a more forgiving and often more realistic indicator (e.g., “haemorrhage” vs. “bleeding”).

---

## Sample Questions & Outputs

Below are samples observed after fine‑tuning (TinyLlama + LoRA on 1–5K examples):

- Q: What adverse event might occur when taking aspirin and warfarin together?
  - A: When aspirin and warfarin are taken together, they may cause **Bone marrow failure**.

- Q: What adverse event is associated with metformin?
  - A: The drug metformin is associated with **Blood bicarbonate decreased**.

- Q: What adverse event might occur when taking ibuprofen and naproxen together?
  - A: When ibuprofen and naproxen are taken together, they may cause **Dysuria**.

> Your results will improve as you increase training size and epochs. Expect additional commentary sometimes (LLMs may append extra sentences). You can post‑process the string to keep only the first sentence or the portion after “Answer:”.

---

## Tips to Improve Quality

- Increase `TRAIN_SIZE` in Cell 3 (e.g., 5,000 → 20,000) if your time allows
- Train for more epochs (Cell 7: `num_train_epochs=3 → 5`)
- Reduce generation randomness during evaluation (`temperature=0.2–0.3`)
- Normalize medical terms (e.g., map synonyms) for accuracy scoring
- Consider using a larger base model if VRAM allows (e.g., Qwen2 7B with CPU offloading)—training will be slower and may require more RAM/VRAM

---

## Reproducibility Notes

- Mixed precision and 4‑bit quantization are used for memory efficiency
- LoRA adapts only a small fraction of parameters (fast and lightweight)
- All paths in the notebook are relative to the project root unless stated
- If you move the project, update the `BASE_DIR` in the notebook accordingly

---

## Troubleshooting

- `CUDA out of memory`: lower `TRAIN_SIZE`, sequence length (256 → 128), or increase `gradient_accumulation_steps`
- `No module named 'llamafactory'`: not used in the final pipeline—this repo uses direct `transformers + peft`
- `Tokenizer mismatch`: always load tokenizer from the adapter folder you saved

---

## License & Intended Use

This project is for research and educational purposes. Drug interaction outputs are generated by an LLM; always verify results with clinical sources before making medical decisions.

---

## Acknowledgements

- Base model: [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- Libraries: Hugging Face `transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`
- Datasets: TwoSides / OffSides
