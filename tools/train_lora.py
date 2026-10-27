"""真实NVFP4 LoRA训练CLI；记录完整逻辑配置，支持优化器边界恢复和执行stop-at。"""

import argparse
import json
from pathlib import Path

from nvfp4lora.train.session import run_session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", default=[])
    parser.add_argument("--evaluation-steps", type=int, nargs="*", default=[])
    parser.add_argument("--eval-corpus", type=Path)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--code-commit", default="uncommitted")
    parser.add_argument("--snapshot-first-micro", action="store_true")
    parser.add_argument("--skip-delta-fingerprint", action="store_true")
    args = parser.parse_args()
    run_session(bundle=args.bundle, data=args.data, settings=json.loads(args.config.read_text(encoding="utf-8-sig")),
                output=args.output, resume=args.resume, stop_at=args.stop_at, checkpoint_steps=args.checkpoint_steps,
                evaluation_steps=args.evaluation_steps, eval_corpus=args.eval_corpus, eval_batch_size=args.eval_batch_size,
                code_commit=args.code_commit, snapshot_first_micro=args.snapshot_first_micro,
                record_delta_w=not args.skip_delta_fingerprint)


if __name__ == "__main__":
    main()
