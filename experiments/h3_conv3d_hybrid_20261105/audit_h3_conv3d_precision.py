"""Controlled NPU precision diagnostic around the unchanged causality audit.

Retains the original failing evidence and original thresholds. Records the
effective accelerator math settings and adds one/three-frame prefix checks.
"""
import argparse
import sys

import torch
import audit_h3_conv3d_causality as audit


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--hf32-conv', choices=('true', 'false'), required=True)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    original_select = audit.select_device
    original_execute = audit.execute
    audit.CUTS = (1, 3, 32, 149, 160)

    def select_device(spec):
        device = original_select(spec)
        if device.type != 'npu':
            raise ValueError('This controlled diagnostic requires an NPU')
        torch.npu.conv.allow_hf32 = options.hf32_conv == 'true'
        torch.npu.matmul.allow_hf32 = False
        return device

    def execute(args, report):
        report['precision_driver'] = audit.fingerprint(__file__)
        report['requested_conv_hf32'] = options.hf32_conv == 'true'
        try:
            original_execute(args, report)
        finally:
            import torch_npu
            report['execution_precision'] = dict(
                torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                conv_allow_hf32=torch.npu.conv.allow_hf32,
                matmul_allow_hf32=torch.npu.matmul.allow_hf32,
                float32_matmul_precision=torch.get_float32_matmul_precision())

    audit.select_device = select_device
    audit.execute = execute
    audit.main()


if __name__ == '__main__':
    main()
