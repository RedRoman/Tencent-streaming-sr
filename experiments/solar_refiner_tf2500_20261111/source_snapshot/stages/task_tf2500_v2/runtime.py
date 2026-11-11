"""Task LoRA with per-block recomputation and CPU storage of frozen weights/activations.

The native SolarRenderer model forward and causal training document are unchanged.
Frozen parameters stay on the device through each recomputed block's backward.
"""
import torch
from torch import nn


class TaskLoRALinear(nn.Module):
    def __init__(self, base, rank, device):
        super().__init__()
        self.base = base
        self.a = nn.Parameter((torch.randn(rank, base.in_features) * .01).to(device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=device))
        self.enabled = True

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        output = self.base(x)
        if self.enabled:
            # alpha=rank, scale=1; BF16 backbone, FP32 trainable low-rank branch.
            output = output + ((x.float() @ self.a.T) @ self.b.T).to(output.dtype)
        return output


class RecomputeBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runtime, index, args, hidden, *params):
        ctx.runtime, ctx.index, ctx.args, ctx.params = runtime, index, args, params
        ctx.save_for_backward(hidden.detach().to('cpu', copy=True))
        return runtime.run_block(index, hidden, args)

    @staticmethod
    def backward(ctx, gradient):
        rt = ctx.runtime
        hidden = ctx.saved_tensors[0].to(rt.device).requires_grad_(True)
        rt.load(ctx.index)
        try:
            with torch.enable_grad():
                output = rt.original[ctx.index](hidden, *ctx.args)
                gradients = torch.autograd.grad(output, (hidden, *ctx.params), gradient,
                                                allow_unused=False)
            return (None, None, None, *gradients)
        finally:
            rt.release(ctx.index)


class TrainingRuntime:
    def __init__(self, model, device, rank=16, resident=0, log=lambda *a, **k: None):
        self.model, self.device = model, torch.device(device)
        self.resident, self.log = resident, log
        self.host, self.original, self.layer_params, self.adapters = {}, [], [], []
        assert all(not p.requires_grad for p in model.parameters())
        model.gradient_checkpointing = False
        for index, block in enumerate(model.transformer_blocks):
            for path in ('attn.to_q', 'attn.to_k', 'attn.to_v', 'attn.to_out.0'):
                parent, key = path.rsplit('.', 1)
                adapter = TaskLoRALinear(block.get_submodule(path), rank, self.device)
                setattr(block.get_submodule(parent), key, adapter)
                self.adapters.append(adapter)
            self.original.append(block.forward)
            self.layer_params.append(tuple(p for p in block.parameters() if p.requires_grad))
            self.host[index] = [(m, k, p.detach()) for m in block.modules()
                                for k, p in m._parameters.items()
                                if p is not None and not p.requires_grad]
            self.host[index] += [(m, k, b.detach()) for m in block.modules()
                                 for k, b in m._buffers.items() if b is not None]
            if index < resident:
                self.load(index)

            def forward(hidden, *args, _index=index, **kwargs):
                if kwargs:
                    raise ValueError('Native block arguments must remain positional')
                if any(isinstance(v, torch.Tensor) and v.requires_grad for v in args):
                    raise ValueError('Only block inputs and task LoRA may carry gradients')
                if torch.is_grad_enabled():
                    return RecomputeBlock.apply(self, _index, args, hidden, *self.layer_params[_index])
                return self.run_block(_index, hidden, args)
            block.forward = forward
        for name, child in model.named_children():
            if name != 'transformer_blocks':
                child.to(self.device)
        self.ordered = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.params = [p for n, p in self.ordered]
        assert len(self.adapters) == 200 and len(self.params) == 400
        assert all(p.dtype == torch.float32 and p.device == self.device for p in self.params)
        log('training_runtime_ready',rank=rank,alpha=rank,adapter_modules=len(self.adapters),
            trainable_parameters=sum(p.numel() for p in self.params),resident_blocks=resident)

    def load(self, index):
        for module, key, host in self.host[index]:
            if key in module._parameters:
                if module._parameters[key].device != self.device:
                    module._parameters[key] = nn.Parameter(host.to(self.device), requires_grad=False)
            elif module._buffers[key].device != self.device:
                module._buffers[key] = host.to(self.device)

    def release(self, index):
        if index < self.resident:
            return
        for module, key, host in self.host[index]:
            if key in module._parameters:
                module._parameters[key] = nn.Parameter(host, requires_grad=False)
            else:
                module._buffers[key] = host

    def run_block(self, index, hidden, args):
        self.load(index)
        try:
            return self.original[index](hidden, *args)
        finally:
            self.release(index)

    def adapter_state(self):
        return {n: p.detach().to('cpu', copy=True) for n, p in self.ordered}

    def load_adapter(self, state):
        assert list(state) == [n for n, p in self.ordered]
        with torch.no_grad():
            for name, param in self.ordered:
                assert state[name].shape == param.shape and state[name].dtype == param.dtype
                param.copy_(state[name].to(param.device))

    def enable_adapters(self, enabled):
        for adapter in self.adapters:
            adapter.enabled = enabled

    def set_resident(self, count):
        """Change frozen-weight placement only, between complete forward/backward steps."""
        assert 0 <= count <= len(self.original)
        self.resident = count
        for index in range(len(self.original)):
            if index < count:
                self.load(index)
            else:
                self.release(index)
        torch.npu.empty_cache()
