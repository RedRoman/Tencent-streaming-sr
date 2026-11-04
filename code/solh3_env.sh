# Sol-H3 NPU environment (host 30778).
#
# The system CANN at /usr/local/Ascend/cann-8.5.2 ships 910B kernel configs for
# ops_oam only; every other operator package (ops_nn/ops_math/ops_cv/
# ops_transformer/ops_legacy) is missing, so any aclnn call fails with 561103
# "Parse dynamic kernel config fail". The private tree below has all five.
#
# Note: CPython teardown prints "free(): invalid pointer" and aborts with
# SIGABRT after main() completes. Computation is unaffected -- end worker
# processes with os._exit(0) and read the real status from their output.

export SOLH3_CANN=/home/ma-user/modelarts/user-job-dir/sol_h3/software/cann-8.5.2-910b-project

export ASCEND_HOME_PATH="$SOLH3_CANN"
export ASCEND_TOOLKIT_HOME="$SOLH3_CANN"
export TOOLKIT_HOME="$SOLH3_CANN"
export ASCEND_OPP_PATH="$SOLH3_CANN/opp"
export ASCEND_AICPU_PATH="$SOLH3_CANN"

export LD_LIBRARY_PATH="$SOLH3_CANN/lib64:$SOLH3_CANN/lib64/plugin/opskernel:$SOLH3_CANN/lib64/plugin/nnengine:$SOLH3_CANN/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
# The private tree supplies kernel binaries but not the `tbe` Python package, so
# the system CANN's site-packages must stay on the path: without it any op that
# needs a dynamic compile dies with "No module named 'tbe'" inside GE init.
export PYTHONPATH="$SOLH3_CANN/python/site-packages:$SOLH3_CANN/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/cann-8.5.2/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages"

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export SOLH3_PY=/home/ma-user/modelarts/user-job-dir/sol_h3/.venv/bin/python
