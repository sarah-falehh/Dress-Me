"""Build a self-contained GPU notebook for LLMAgent only; no account data."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = (ROOT / "app/services/llm_conversation.py").read_text(encoding="utf-8")
    data = {split: [json.loads(line) for line in (ROOT / f"app/data/llm_agent_sft/{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            for split in ("train", "validation", "test")}
    cells = []

    def cell(kind, content):
        item = {"cell_type": kind, "metadata": {}, "source": content.splitlines(keepends=True), "id": f"llm-agent-{len(cells)}"}
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", """# DressMe — LoRA du LLMAgent uniquement

1. Ouvrir ce fichier dans Google Colab.
2. Choisir **Exécution → Modifier le type d'exécution → GPU**, puis connecter le runtime.
3. Exécuter les cellules dans l'ordre et conserver le ZIP téléchargé à la fin.

Le modèle de base est **Qwen/Qwen2.5-0.5B-Instruct**. Ses poids sont gelés ;
seuls les adaptateurs LoRA sont entraînés. Ce n'est pas un entraînement du
GPT-OSS hébergé sur Groq. Aucun autre agent et aucune configuration du projet
ne sont modifiés. L'adaptateur n'est pas activé automatiquement dans l'application.

**Dataset pilote synthétique : 20 train, 4 validation, 4 test.** Les scénarios sont
séparés avant paraphrase. Relire les exemples ci-dessous ; ce petit jeu n'est pas
un benchmark de qualité et ne suffit pas à démontrer une généralisation.
Aucun entraînement n'a encore eu lieu dans ce notebook : ses sorties sont vides.
La disponibilité d'un GPU gratuit dépend de Colab ; aucun achat n'est nécessaire
dans ce workflow.

Références : [PEFT](https://huggingface.co/docs/peft/quicktour),
[modèle Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
""")
    cell("code", """import os, sys, subprocess
os.environ['USE_TF'] = '0'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'transformers==4.57.6', 'peft==0.18.1', 'datasets==4.5.0',
    'accelerate==1.14.0', 'pydantic>=2,<3'])
# Utiliser le PyTorch CUDA du runtime Colab ; ne pas le remplacer par une roue CPU.
import torch
assert torch.cuda.is_available(), 'Active un runtime GPU dans le menu Exécution.'
print('GPU :', torch.cuda.get_device_name(0))
print('Mémoire GPU :', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'Go')
""")
    cell("code", """from pathlib import Path
import json, hashlib, random, time
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
EPOCHS = 3
LEARNING_RATE = 2e-4
SEED = 42
MAX_LENGTH = 4096
OUTPUT = Path('/content/llm_agent_lora')
if OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise RuntimeError('Résultats déjà présents : change OUTPUT pour préserver cet entraînement.')
set_seed(SEED)
""")
    cell("code", f"DATA = json.loads({json.dumps(json.dumps(data, ensure_ascii=False), ensure_ascii=False)})\n"
         f"VALIDATOR_SOURCE = {source!r}\n"
         "# Copie autonome du validateur de LLMAgent, sans importer d'autres agents.\n"
         "namespace = {'__name__': '__main__'}\nexec(compile(VALIDATOR_SOURCE, 'llm_conversation.py', 'exec'), namespace)\n"
         "build_plan = namespace['build_conversation_plan']\n"
         "groups = {s: {r['group'] for r in rows} for s, rows in DATA.items()}\n"
         "assert not (groups['train'] & groups['validation'] or groups['train'] & groups['test'] or groups['validation'] & groups['test'])\n"
         "for split, rows in DATA.items():\n"
         "    for row in rows:\n        build_plan(row['facts']).render(row['messages'][-1]['content'])\n"
         "    print(split, len(rows), 'exemples valides')\n"
         "for row in DATA['train']:\n    print(row['facts']['question'])\n    print(row['messages'][-1]['content'], '\\n')\n")
    cell("markdown", """## Préparation de la loss

La loss est calculée uniquement sur la dernière réponse assistant. Les instructions,
le contexte et le padding reçoivent le label `-100`. Un exemple dépassant la limite
est refusé, sans troncature silencieuse de sa réponse.
""")
    cell("code", """tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

def encode(row):
    prefix = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=False)
    assert full[:len(prefix)] == prefix, 'Préfixe du chat template incompatible'
    assert len(prefix) < len(full) <= MAX_LENGTH, 'Longueur invalide : ne pas tronquer la cible'
    return {'input_ids': full, 'attention_mask': [1] * len(full),
            'labels': [-100] * len(prefix) + full[len(prefix):]}

datasets = {s: Dataset.from_list([encode(row) for row in rows]) for s, rows in DATA.items()}

def collate(batch):
    length = max(len(row['input_ids']) for row in batch)
    return {key: torch.tensor([row[key] + [padding] * (length-len(row[key])) for row in batch])
            for key, padding in [('input_ids', tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100)]}

base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
base.config.use_cache = False
model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM,
    r=8, lora_alpha=16, lora_dropout=0.05, bias='none', target_modules=['q_proj', 'v_proj']))
model.enable_input_require_grads()
model.print_trainable_parameters()
initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
assert initial and all('lora_' in name for name in initial), 'Seuls les paramètres LoRA doivent être entraînés'
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
""")
    cell("code", """training_args = TrainingArguments(
    output_dir=str(OUTPUT / 'checkpoints'), num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE, per_device_train_batch_size=1,
    per_device_eval_batch_size=1, gradient_accumulation_steps=4,
    gradient_checkpointing=True, fp16=True, bf16=False,
    eval_strategy='epoch', save_strategy='epoch', save_total_limit=1,
    logging_steps=1, report_to=[], seed=SEED)
trainer = Trainer(model=model, args=training_args, train_dataset=datasets['train'],
    eval_dataset=datasets['validation'], data_collator=collate)

# L'adaptateur initial est équivalent à la base : ses matrices B sont initialisées à zéro.
before = trainer.evaluate(datasets['test'], metric_key_prefix='test_before')
started = time.monotonic()
training_result = trainer.train()
training_seconds = time.monotonic() - started
after = trainer.evaluate(datasets['test'], metric_key_prefix='test_after')
model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)
changed = sum(not torch.equal(initial[name], p.detach().cpu())
              for name, p in model.named_parameters() if name in initial)
assert changed > 0, 'Aucun paramètre LoRA modifié : entraînement à vérifier'
report = {'base_model': MODEL_ID, 'base_revision': getattr(model.config, '_commit_hash', None),
    'scope': 'LLMAgent conversation only', 'dataset_status': 'small_synthetic_pilot',
    'counts': {s: len(rows) for s, rows in DATA.items()},
    'dataset_sha256': hashlib.sha256(json.dumps(DATA, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
    'trainable_parameters': trainable, 'total_parameters': total,
    'changed_adapter_tensors': changed, 'lora': {'r': 8, 'alpha': 16, 'dropout': 0.05, 'targets': ['q_proj', 'v_proj']},
    'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'batch_size': 1, 'gradient_accumulation': 4,
    'seed': SEED, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
    'training_seconds': training_seconds, 'training': training_result.metrics,
    'before': before, 'after': after, 'log_history': trainer.state.log_history,
    'application_activation': False}
(OUTPUT / 'training_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
""")
    cell("markdown", """## Comparaison des réponses générées

Les mêmes questions de test sont générées avec et sans adaptateur, en mode déterministe.
Le validateur contrôle le JSON, les références et les règles existantes. Un JSON valide
ne prouve pas la vérité de chaque phrase ; relire les réponses avant de conclure.
""")
    cell("code", """from contextlib import nullcontext
model.eval()
model.config.use_cache = True
records = []
for row in DATA['test']:
    prompt = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True,
        add_generation_prompt=True, return_tensors='pt').to(model.device)
    for adapted in (False, True):
        started = time.monotonic()
        with (nullcontext() if adapted else model.disable_adapter()), torch.inference_mode():
            output = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(output[0, prompt.shape[-1]:], skip_special_tokens=True)
        error = None
        try:
            build_plan(row['facts']).render(text)
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
        record = {'question': row['facts']['question'], 'group': row['group'],
            'adapter_enabled': adapted, 'output': text, 'runtime_valid': error is None,
            'validation_error': error, 'seconds': time.monotonic()-started}
        records.append(record)
        print('LoRA' if adapted else 'BASE', record['question'], '\\n', text, '\\nValid:', error is None)
(OUTPUT / 'generation_evaluation.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
""")
    cell("code", """# Télécharger les preuves et l'adaptateur avant de fermer Colab.
import shutil
from google.colab import files
for split, rows in DATA.items():
    (OUTPUT / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\\n' for r in rows), encoding='utf-8')
zip_path = shutil.make_archive('/content/llm_agent_lora_results', 'zip', OUTPUT)
files.download(zip_path)
""")
    cell("markdown", """## Pour l'oral

Après une exécution réussie seulement : « J'ai adapté Qwen avec LoRA pour les
réponses de LLMAgent. Les poids de base sont gelés ; j'entraîne les matrices de
rang 8 sur les projections de requête et de valeur. »

Montrer `adapter_config.json`, `adapter_model.safetensors`, `training_report.json`
et `generation_evaluation.json`. Présenter les vrais résultats, y compris les erreurs.
Ne pas annoncer une amélioration générale à partir de ce seul petit pilote.
L'application continue à utiliser sa configuration actuelle tant qu'une intégration
explicite de l'adaptateur n'a pas été faite.
""")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "accelerator": "GPU", "colab": {"name": "LLMAgent_LoRA_GPU.ipynb"}}}
    out = ROOT / "LLMAgent_LoRA_GPU.ipynb"
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
