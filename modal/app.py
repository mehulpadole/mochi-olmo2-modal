"""OLMo 2 7B Base serving app for MoCHi's Tomio Labs catalog.

The official float32 Hugging Face checkpoint is kept unchanged in a dedicated
Modal Volume. At startup, Transformers loads it as bfloat16 on one L4. The
OpenAI-compatible adapter uses OLMo 2's role markers, but this remains a base
model and is not instruction-tuned.
"""

import hashlib
import hmac
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid

import modal


APP_NAME = "mochi-olmo2-7b-base"
VOLUME_NAME = "olmo2-7b-base-models"
SECRET_NAME = "olmo2-7b-base-api"
MODEL_MOUNT = "/models"
MODEL_PATH = f"{MODEL_MOUNT}/olmo-2-1124-7b"
MODEL_REPOSITORY = "allenai/OLMo-2-1124-7B"
MODEL_REVISION = "7df9a82518afdecae4e8c026b27adccc8c1f0032"
MODEL_WEIGHT_FILES = tuple(
    f"model-{index:05d}-of-00006.safetensors" for index in range(1, 7)
)
CONTEXT_SIZE = 4096
MAX_OUTPUT_TOKENS = 2048

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub==0.33.4")
)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.53.3",
        "accelerate==1.9.0",
        "safetensors==0.5.3",
        "huggingface_hub==0.33.4",
        "fastapi==0.115.14",
        "uvicorn==0.34.3",
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_and_verify_model() -> dict[str, object]:
    from huggingface_hub import snapshot_download

    model_directory = Path(MODEL_PATH)
    model_directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        local_dir=str(model_directory),
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.model",
            "*.safetensors",
            "merges.txt",
            "vocab.json",
        ],
    )

    weights: list[dict[str, object]] = []
    for filename in MODEL_WEIGHT_FILES:
        path = model_directory / filename
        if not path.is_file():
            raise RuntimeError(f"Pinned OLMo 2 checkpoint is missing {filename}.")
        weights.append(
            {
                "filename": filename,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    manifest = {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "weights": weights,
    }
    manifest_path = model_directory / "mochi-model-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _verify_volume_manifest() -> None:
    model_directory = Path(MODEL_PATH)
    manifest_path = model_directory / "mochi-model-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("OLMo 2 model manifest is missing; run the preload step.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("repository") != MODEL_REPOSITORY
        or manifest.get("revision") != MODEL_REVISION
    ):
        raise RuntimeError("The Modal Volume contains an unexpected OLMo 2 revision.")

    verified_files = {
        item.get("filename"): item for item in manifest.get("weights", [])
    }
    for filename in MODEL_WEIGHT_FILES:
        path = model_directory / filename
        recorded = verified_files.get(filename)
        if not path.is_file() or not recorded:
            raise RuntimeError(f"The verified OLMo 2 artifact is missing {filename}.")
        if path.stat().st_size != recorded.get("size_bytes"):
            raise RuntimeError(f"The OLMo 2 artifact size changed for {filename}.")


@app.function(
    image=download_image,
    volumes={MODEL_MOUNT: model_volume},
    timeout=3 * 60 * 60,
    retries=0,
)
def preload_model() -> dict[str, object]:
    """Download the immutable source revision and record each shard's SHA-256."""

    manifest = _download_and_verify_model()
    model_volume.commit()
    total_bytes = sum(int(item["size_bytes"]) for item in manifest["weights"])
    print(
        "OLMo 2 7B Base verified:",
        manifest["repository"],
        manifest["revision"],
        len(manifest["weights"]),
        "weight shards,",
        total_bytes,
        "bytes",
    )
    return manifest


def _verify_l4_and_bf16(torch) -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("CUDA/L4 verification failed; refusing CPU fallback.") from error

    gpu_info = result.stdout.strip()
    if "L4" not in gpu_info:
        raise RuntimeError(f"Expected an NVIDIA L4 GPU, got: {gpu_info or 'none'}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The L4 CUDA runtime does not support the required bfloat16 mode.")

    print(f"OLMo 2 GPU verified: {gpu_info}; bfloat16 supported")
    return gpu_info


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in ("text", "input_text")
        )
    return ""


def _serialize_messages(messages: list[dict[str, object]], tokenizer) -> str:
    parts = [tokenizer.bos_token or ""]
    valid_roles = {"system", "user", "assistant"}
    for item in messages:
        role = item.get("role")
        if role not in valid_roles:
            continue
        content = _message_text(item.get("content", ""))
        # The base checkpoint tokenizer includes these native message-boundary tokens.
        parts.extend((f"<|im_start|>{role}\n", content, "<|im_end|>\n"))

    if not messages or messages[-1].get("role") != "assistant":
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _bounded_generation_options(payload: dict[str, object], input_tokens: int) -> dict[str, object]:
    requested = payload.get("max_completion_tokens", payload.get("max_tokens", 512))
    try:
        requested_tokens = int(requested)
    except (TypeError, ValueError):
        requested_tokens = 512

    max_new_tokens = max(
        1,
        min(MAX_OUTPUT_TOKENS, requested_tokens, CONTEXT_SIZE - input_tokens),
    )
    try:
        temperature = float(payload.get("temperature", 0.7))
    except (TypeError, ValueError):
        temperature = 0.7
    try:
        top_p = float(payload.get("top_p", 0.95))
    except (TypeError, ValueError):
        top_p = 0.95
    try:
        top_k = int(payload.get("top_k", 50))
    except (TypeError, ValueError):
        top_k = 50
    try:
        repetition_penalty = float(
            payload.get("repetition_penalty", payload.get("repeat_penalty", 1.0))
        )
    except (TypeError, ValueError):
        repetition_penalty = 1.0

    temperature = min(2.0, max(0.0, temperature))
    top_p = min(1.0, max(0.01, top_p))
    top_k = min(256, max(1, top_k))
    repetition_penalty = min(2.0, max(1.0, repetition_penalty))
    options: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "use_cache": True,
    }
    if temperature > 0:
        options["temperature"] = temperature
    return options


@app.function(
    image=image,
    gpu="L4",
    volumes={MODEL_MOUNT: model_volume},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    min_containers=0,
    max_containers=1,
    scaledown_window=15 * 60,
    timeout=60 * 60,
)
@modal.asgi_app()
def openai_server():
    """Serve FastAPI directly as an authenticated Modal ASGI endpoint."""

    import torch
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
        TextIteratorStreamer,
    )

    _verify_volume_manifest()
    gpu_info = _verify_l4_and_bf16(torch)
    api_key = os.environ.get("OLMO2_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OLMO2_API_KEY is missing from the Modal Secret.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model.eval()
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("Not all OLMo 2 weights were placed on the L4 GPU.")

    allocated_gb = torch.cuda.memory_allocated() / (1024**3)
    reserved_gb = torch.cuda.memory_reserved() / (1024**3)
    print(
        f"OLMo 2 loaded fully on L4 as bfloat16; context={CONTEXT_SIZE}; "
        f"VRAM allocated={allocated_gb:.2f} GiB, reserved={reserved_gb:.2f} GiB"
    )

    generation_lock = threading.Lock()
    api = FastAPI()
    stop_token_ids = {
        token_id
        for token_id in (
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
        if isinstance(token_id, int) and token_id >= 0
    }
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    def authorized(request: Request) -> bool:
        authorization = request.headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(token.strip(), api_key)

    def encode_prompt(messages: list[dict[str, object]], max_output: int):
        text = _serialize_messages(messages, tokenizer)
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"]
        input_limit = max(1, CONTEXT_SIZE - max_output)
        if input_ids.shape[-1] > input_limit:
            input_ids = input_ids[:, -input_limit:]
        input_ids = input_ids.to("cuda:0")
        return input_ids, torch.ones_like(input_ids)

    class StopIfCancelled(StoppingCriteria):
        def __init__(self, cancellation: threading.Event):
            self.cancellation = cancellation

        def __call__(self, input_ids, scores, **kwargs):
            return self.cancellation.is_set()

    @api.get("/health")
    def health():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "status": "ok",
            "model": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "gpu": gpu_info.split(",", maxsplit=1)[0].strip(),
            "context_length": CONTEXT_SIZE,
            "vram_total_bytes": total_bytes,
            "vram_free_bytes": free_bytes,
            "bfloat16": True,
        }

    @api.get("/v1/models")
    def list_models(request: Request):
        if not authorized(request):
            return JSONResponse({"error": {"message": "Unauthorized"}}, status_code=401)
        return {"object": "list", "data": [{"id": "olmo-2-7b-base", "object": "model"}]}

    @api.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        if not authorized(request):
            return JSONResponse({"error": {"message": "Unauthorized"}}, status_code=401)

        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": {"message": "Invalid JSON body"}}, status_code=400)

        messages = payload.get("messages") if isinstance(payload, dict) else None
        if not isinstance(messages, list) or not messages:
            return JSONResponse(
                {"error": {"message": "At least one chat message is required"}},
                status_code=400,
            )

        output_requested = payload.get(
            "max_completion_tokens", payload.get("max_tokens", 512)
        )
        try:
            output_requested = int(output_requested)
        except (TypeError, ValueError):
            output_requested = 512
        requested_output = min(MAX_OUTPUT_TOKENS, max(1, output_requested))
        input_ids, attention_mask = encode_prompt(messages, requested_output)
        options = _bounded_generation_options(payload, int(input_ids.shape[-1]))
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        returned_model = str(payload.get("model") or "olmo-2-7b-base")

        if not payload.get("stream", False):
            with generation_lock, torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    eos_token_id=sorted(stop_token_ids),
                    pad_token_id=pad_token_id,
                    **options,
                )
            completion_tokens = int(output_ids.shape[-1] - input_ids.shape[-1])
            answer = tokenizer.decode(
                output_ids[0, input_ids.shape[-1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": returned_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "length"
                        if completion_tokens >= int(options["max_new_tokens"])
                        else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": int(input_ids.shape[-1]),
                    "completion_tokens": completion_tokens,
                    "total_tokens": int(input_ids.shape[-1]) + completion_tokens,
                },
            }

        def stream_response():
            with generation_lock:
                cancellation = threading.Event()
                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                    timeout=0.5,
                )
                errors: list[BaseException] = []

                def generate():
                    try:
                        with torch.inference_mode():
                            model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                eos_token_id=sorted(stop_token_ids),
                                pad_token_id=pad_token_id,
                                streamer=streamer,
                                stopping_criteria=StoppingCriteriaList(
                                    [StopIfCancelled(cancellation)]
                                ),
                                **options,
                            )
                    except BaseException as error:  # propagated to the SSE consumer
                        errors.append(error)

                worker = threading.Thread(target=generate, daemon=True)
                worker.start()
                yield _sse(
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": returned_model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                )

                try:
                    while True:
                        try:
                            text = next(streamer)
                        except queue.Empty:
                            if not worker.is_alive():
                                break
                            continue
                        except StopIteration:
                            break
                        if text:
                            yield _sse(
                                {
                                    "id": response_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": returned_model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": text},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                    worker.join(timeout=1)
                    if errors:
                        yield _sse(
                            {
                                "error": {
                                    "message": f"OLMo 2 generation failed ({type(errors[0]).__name__})."
                                }
                            }
                        )
                    else:
                        yield _sse(
                            {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": returned_model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            }
                        )
                        yield "data: [DONE]\n\n"
                finally:
                    cancellation.set()
                    worker.join(timeout=10)

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    return api


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.local_entrypoint()
def main() -> None:
    """`modal run modal/olmo2/app.py` preloads the pinned model into its Volume."""

    result = preload_model.remote()
    print(
        "OLMo 2 preload complete:",
        result["repository"],
        result["revision"],
        len(result["weights"]),
        "shards verified",
    )
