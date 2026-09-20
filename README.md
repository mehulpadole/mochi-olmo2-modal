# OLMo 2 7B Base on Modal

Standalone Modal deployment source for MoCHi's server-managed OLMo 2 7B Base
model. The official pinned Hugging Face checkpoint is loaded as bfloat16 on one
NVIDIA L4 through Transformers and exposed through an authenticated
OpenAI-compatible streaming endpoint.

- Modal app: `mochi-olmo2-7b-base`
- Volume: `olmo2-7b-base-models`
- Modal Secret: `olmo2-7b-base-api`, containing `OLMO2_API_KEY`
- Context: 4K; maximum output: 2K
- Scaling: `min_containers=0`, `max_containers=1`, 15-minute scale-down

Model shards are downloaded and recorded with SHA-256 checksums in the Modal
Volume by the preload step. They are not stored in Git. Credentials and
MoCHi/Cloudflare secrets are also never stored here.

```bash
modal run modal/app.py
modal deploy modal/app.py
```

This is a base checkpoint rather than an instruction-tuned chat checkpoint, so
repetitive or instruction-following failures are expected. Configure the
resulting endpoint and matching server-side key in MoCHi's Cloudflare Worker.
Never expose the key in browser code and do not add a recurring health poller.
