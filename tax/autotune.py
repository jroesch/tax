# @create_backend
# def tvm(subgraph):
#     return subgraph.wrap_returns(
#         tvm_compile_inner(
#             subgraph.scripted,
#             subgraph.example_inputs,
#             tuning_option=None,
#             cuda=subgraph.is_cuda,
#         )
#     )


# @create_backend
# def ansor(subgraph):
#     """
#     WARNING: this backend takes hours or days to train and
#     often produces a slower result than the default schedule.
#     """
#     return subgraph.wrap_returns(
#         tvm_compile_inner(
#             subgraph.scripted,
#             subgraph.example_inputs,
#             tuning_option="auto_scheduler",
#             log_file=subgraph.filename("ansor"),
#             cuda=subgraph.is_cuda,
#         )
#     )


# @create_backend
# def tvm_meta_schedule(subgraph):
#     return subgraph.wrap_returns(
#         tvm_compile_inner(
#             subgraph.scripted,
#             subgraph.example_inputs,
#             tuning_option="meta_schedule",
#             trials=20000,
#             cuda=subgraph.is_cuda,
#         )
#     )


# @functools.lru_cache(None)
# def llvm_target():
#     if "avx512" in open("/proc/cpuinfo").read():
#         return "llvm -mcpu=skylake-avx512"
#     return "llvm -mcpu=core-avx2"


def tvm_compile_inner(
    jit_mod, example_inputs, tuning_option=None, log_file=None, trials=20000, cuda=False
):
    try:
        import tvm
        from tvm import relay
        from tvm.contrib import graph_executor

        shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
        mod, params = relay.frontend.from_pytorch(jit_mod, shape_list)
        if cuda:
            dev = tvm.cuda(0)
            target = tvm.target.cuda()
        else:
            dev = tvm.cpu(0)
            target = tvm.target.Target(llvm_target())

        if tuning_option == "auto_scheduler":
            from tvm import auto_scheduler

            if log_file is None:
                log_file = tempfile.NamedTemporaryFile()
            if not os.path.exists(log_file):
                tasks, task_weights = auto_scheduler.extract_tasks(
                    mod["main"], params, target
                )
                for task in tasks:
                    print(task.compute_dag)
                else:
                    print("No tasks")
                if len(tasks) != 0:
                    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
                    if not os.path.exists(log_file):
                        assert trials > 0
                        tune_option = auto_scheduler.TuningOptions(
                            num_measure_trials=trials,
                            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                            early_stopping=2000,
                        )
                        try:
                            tuner.tune(tune_option)
                        except Exception:
                            if os.path.exists(log_file):
                                os.unlink(log_file)
                            raise

            with auto_scheduler.ApplyHistoryBest(log_file):
                with tvm.transform.PassContext(
                    opt_level=3, config={"relay.backend.use_auto_scheduler": True}
                ):
                    lib = relay.build(mod, target=target, params=params)
        elif tuning_option == "meta_schedule":
            from os import path as osp

            from tvm.meta_schedule import TuneConfig
            from tvm.meta_schedule.database import JSONDatabase
            from tvm.meta_schedule.tune import tune_relay

            with tempfile.TemporaryDirectory() as work_dir:
                if log_file is not None:
                    assert osp.isdir(
                        log_file
                    ), "TVM's meta_schedule requires a directory for storing log files."
                    work_dir = log_file
                lib: tvm.runtime.Module = tune_relay(
                    mod=mod,
                    params=params,
                    target=target,
                    config=TuneConfig(
                        strategy="evolutionary",
                        num_trials_per_iter=32,
                        max_trials_per_task=32,
                        max_trials_global=trials,
                    ),
                    work_dir=work_dir,
                    database=JSONDatabase(
                        osp.join(work_dir, "workload.json"),
                        osp.join(work_dir, "records.json"),
                    ),
                )
        elif tuning_option is None:
            # no autotuning (for debugging)
            with tvm.transform.PassContext(opt_level=10):
                lib = relay.build(mod, target=target, params=params)
        else:
            raise NotImplementedError(
                "This tuning option is invalid/not implemented for torchdynamo's TVM-related backend. "
                "There are three available options including None, auto_scheduler and meta_schedule."
            )

        m = graph_executor.GraphModule(lib["default"](dev))

        def to_torch_tensor(nd_tensor):
            """A helper function to transfer a NDArray to torch.tensor."""
            if nd_tensor.dtype == "bool":
                # DLPack does not support boolean so it can't be handled by
                # torch.utils.dlpack.from_pack. Workaround by going through
                # numpy, although this brings additional data copy overhead.
                return torch.from_numpy(nd_tensor.numpy())
            return torch.utils.dlpack.from_dlpack(nd_tensor.to_dlpack())

        def exec_tvm(*args):
            args = [a.contiguous() for a in args]
            for idx, arg in enumerate(args, 0):
                if arg.dim() != 0:
                    if arg.requires_grad:
                        arg = arg.detach()
                    m.set_input(
                        f"inp_{idx}",
                        tvm.nd.array(arg.numpy(), dev),
                    )
            m.run()
            return [
                to_torch_tensor(m.get_output(i)) for i in range(m.get_num_outputs())
            ]

        return exec_tvm
    except Exception:
        log.exception("tvm error")
        return jit_mod  # explicit fall back to eager
