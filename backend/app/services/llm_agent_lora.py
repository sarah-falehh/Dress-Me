"""Local inference for the Colab-trained LLMAgent conversation adapter only."""
from functools import lru_cache
import json
import os
from pathlib import Path
import re
from threading import RLock

from app.core.config import Settings
from app.services.llm_conversation import ConversationPlan, ConversationValidationError

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_generation_lock = RLock()


def adapter_identity(directory: str) -> tuple[str, int, int]:
    root = Path(directory).resolve(strict=True)
    weights = root / "adapter_model.safetensors"
    stat = weights.stat()
    return str(root), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=1)
def _load_runtime(identity: tuple[str, int, int]):
    # Optional dependencies remain lazy: hosted/default mode does not import PEFT.
    os.environ.setdefault("USE_TF", "0")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(identity[0])
    config = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
    report = json.loads((root / "training_report.json").read_text(encoding="utf-8"))
    revision = report.get("base_revision", "")
    if config.get("base_model_name_or_path") != BASE_MODEL or report.get("base_model") != BASE_MODEL:
        raise ValueError("Unexpected base model for LLMAgent adapter")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Missing pinned base revision")
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=revision,
        local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model = PeftModel.from_pretrained(base, root, is_trainable=False).to(device).eval()
    model.config.use_cache = True
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return model, tokenizer


def _generate(messages: list[dict[str, str]], settings: Settings, schema: dict | None = None) -> str:
    import torch
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

    model, tokenizer = _load_runtime(adapter_identity(settings.llm_lora_adapter_dir))
    inputs = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, return_tensors="pt").to(model.device)
    if inputs.shape[-1] > 8192:
        raise ValueError("Conversation exceeds local input budget")
    prefix = build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(schema))
    with torch.inference_mode():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
            max_new_tokens=settings.llm_lora_max_new_tokens, do_sample=False,
            prefix_allowed_tokens_fn=prefix,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, inputs.shape[-1]:], skip_special_tokens=True)


def generate_conversation(plan: ConversationPlan, payload: str, settings: Settings) -> str:
    messages = plan.messages(payload)
    question = json.loads(payload)["facts"]["question"]
    references = ", ".join(sorted(plan.known_refs)) or "aucune"
    messages[0] = {**messages[0], "content": messages[0]["content"] +
        "\nRéférences permises dans item_refs : " + references + ". " +
        ("Toutes les listes item_refs doivent être vides. " if not plan.known_refs else "") +
        "Ne recopie jamais une consigne technique ou un message de validation dans answer."}
    # One process-wide lock avoids concurrent model loading/generation on the GPU.
    with _generation_lock:
        for attempt in range(2):
            content = _generate(messages, settings, plan.response_format()["json_schema"]["schema"])
            try:
                rendered = plan.render(content)
                answer = json.loads(rendered)["answer"].casefold()
                if any(marker in answer for marker in ("références autorisées", "référence de vêtement n'existe",
                        "respecte le schéma json", "corrige ta réponse", "json demandé")):
                    raise ValueError("Internal repair instructions echoed as advice")
                return rendered
            except (ValueError, TypeError, KeyError) as exc:
                if attempt == 1:
                    raise
                correction = str(exc) if isinstance(exc, ConversationValidationError) else "Réponds uniquement avec le JSON demandé, sans texte autour."
                messages = [*messages, {"role": "assistant", "content": content},
                    {"role": "user", "content": "Ta sortie doit être corrigée : " + correction +
                     " Ne recopie pas ces consignes dans answer. Réponds à la question de style initiale : " + question}]
    raise ValueError("No valid local response")
