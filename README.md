# MoCHi OLMo 2 Modal deployment

Standalone Modal deployment for the official OLMo 2 7B base checkpoint. It is exposed under MoCHi's Tomio Labs provider as an experimental model. This is a base language model, not an instruction-tuned chat checkpoint, so conversational quality and instruction following are not guaranteed.

## What is implemented

- Downloads the pinned Hugging Face checkpoint into a persistent Modal Volume during preload.
- Restricts the download to the model files needed by the Transformers runtime and records per-file size and SHA-256 values in a Volume manifest.
- Loads the checkpoint in bfloat16 on an NVIDIA L4 and verifies that all model parameters are on CUDA.
- Provides authenticated `/health`, `/v1/models`, and `/v1/chat/completions` routes through a small FastAPI ASGI application.
- Supports OpenAI-compatible JSON responses and Server-Sent Event streaming with `"stream": true`.
- Clamps the request context and output length to the deployment's configured limits.

## Deployment configuration

| Setting | Value |
| --- | --- |
| Modal app | `mochi-olmo2-7b-base` |
| Modal Volume | `olmo2-7b-base-models` |
| Modal Secret | `olmo2-7b-base-api` |
| Model source | `allenai/OLMo-2-1124-7B` |
| Source revision | `7df9a82518afdecae4e8c026b27adccc8c1f0032` |
| Runtime | Hugging Face Transformers / PyTorch |
| Precision | bfloat16 |
| GPU | NVIDIA L4 |
| Context / maximum output | 4,096 / 2,048 tokens |
| Scaling | `min_containers=0`, `max_containers=1`, 15-minute scaledown window |

The six safetensor shards are downloaded from the pinned source revision. The resulting manifest is checked during server startup so an unexpected Volume revision is rejected.

## API

After deployment, use the HTTPS URL printed by Modal as `<modal-endpoint>`.

- `GET /health` returns model revision, GPU, context, and memory status after the model is loaded.
- `GET /v1/models` lists `olmo-2-7b-base` and requires a Bearer key.
- `POST /v1/chat/completions` accepts OpenAI-compatible chat payloads and requires a Bearer key.
- Set `"stream": true` to receive incremental Server-Sent Events.
- The server returns `401 Unauthorized` when the Bearer key is missing or incorrect.

Example request shape:

```json
{
  "model": "olmo-2-7b-base",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true
}
```

## Request flow

```text
MoCHi client
  -> MoCHi server-side provider route
  -> authenticated Modal HTTPS endpoint
  -> FastAPI authentication and request adapter
  -> Transformers model on the L4 GPU
  -> OpenAI-compatible response or SSE stream
```

The API key is injected by the Modal Secret and used only inside the server process. It must not be put in browser-visible environment variables or client JavaScript.

## Run and deploy

Install and authenticate Modal, then create the named Secret through a local secret-management workflow. The variable expected by the server is `OLMO2_API_KEY`; do not commit its value.

```bash
python -m pip install --upgrade modal
modal setup
modal secret create olmo2-7b-base-api OLMO2_API_KEY="<value supplied securely>"
modal run modal/app.py
modal deploy modal/app.py
```

`modal run` invokes the preload function and populates the persistent Volume. `modal deploy` publishes the authenticated ASGI endpoint. The serving container refuses to start if the pinned checkpoint manifest is missing or inconsistent.

## Observed benchmark notes

An earlier deployment verification recorded approximately 125.7 seconds for a cold end-to-end request, including about 66.9 seconds of Modal execution and model startup. A warm streaming request began producing output after approximately 8.8 seconds; one stopped test took about 48.3 seconds overall. These are single-run observations, not capacity benchmarks, and the warm run was stopped from the client, so upstream cancellation was not conclusively measured.

The base checkpoint produced repetitive/off-topic text in that test. That behavior is an expected limitation of using a base model for chat and should not be presented as a tuned assistant benchmark.

## Security and release notes

- No API keys, Modal URLs, user data, or model weights are tracked.
- `.env.example` documents the expected variable without containing a secret and is the only `.env*` file allowed by `.gitignore`.
- Keep any private-beta authorization in MoCHi's server-side provider route.
- Review the upstream OLMo license and redistribution terms before making this repository or service public.
