from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class BenchNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1024, 8192),
            torch.nn.GELU(),
            torch.nn.Linear(8192, 8192),
            torch.nn.GELU(),
            torch.nn.Linear(8192, 1024),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_float32_matmul_precision("high")
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)

    torch.manual_seed(1234 + local_rank)
    model = DDP(BenchNet().to(device), device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(args.batch_size, 1024, device=device)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs).square().mean()
        loss.backward()
        optimizer.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    dist.barrier()
    started = time.perf_counter()
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - started

    if dist.get_rank() == 0:
        step_ms = elapsed * 1000 / args.iters
        samples_s = args.batch_size * dist.get_world_size() * args.iters / elapsed
        print(
            f"ddp_training: world_size={dist.get_world_size()} batch_per_gpu={args.batch_size} "
            f"iters={args.iters} step_ms={step_ms:.2f} samples_per_s={samples_s:.1f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
