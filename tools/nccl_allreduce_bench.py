from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist


def bench(label: str, tensors: list[torch.Tensor], iterations: int) -> None:
    for _ in range(2):
        for tensor in tensors:
            dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()
    started = time.perf_counter()
    for _ in range(iterations):
        for tensor in tensors:
            dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - started
    payload = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    if dist.get_rank() == 0:
        print(
            f"{label}: payload_mib={payload / 2**20:.0f} collectives={len(tensors)} "
            f"iterations={iterations} ms_per_iter={elapsed * 1000 / iterations:.2f} "
            f"payload_gib_s={payload * iterations / elapsed / 2**30:.3f}",
            flush=True,
        )


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    device = torch.device(f"cuda:{local_rank}")
    bench("one_64mib", [torch.ones(16 * 2**20, device=device)], 5)
    bench("sixtyfour_1mib", [torch.ones(2**18, device=device) for _ in range(64)], 3)
    bench("one_256mib", [torch.ones(64 * 2**20, device=device)], 3)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
