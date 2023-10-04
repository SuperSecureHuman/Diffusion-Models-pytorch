```sh
╰─ python ddpm_conditional.py                                                                                                                                                                                                                    ─╯
04:31:40 - INFO: Starting epoch 0:
  0%|                                                                                                                                                                                                                      | 0/2000 [00:00<?, ?it/s][2023-10-04 16:31:40,313] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo start tracing forward
[2023-10-04 16:31:40,320] torch._dynamo.output_graph: [INFO] Step 2: calling compiler function debug_wrapper
[2023-10-04 16:31:41,213] torch._inductor.compile_fx: [INFO] Step 3: torchinductor compiling FORWARDS graph 0
[2023-10-04 16:31:41,281] torch._inductor.compile_fx: [INFO] Step 3: torchinductor done compiling FORWARDS graph 0
[2023-10-04 16:31:41,282] torch._dynamo.output_graph: [INFO] Step 2: done compiler function debug_wrapper
[2023-10-04 16:31:41,426] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo start tracing pos_encoding
[2023-10-04 16:31:41,430] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo start tracing <graph break in pos_encoding>
[2023-10-04 16:31:41,437] torch._dynamo.output_graph: [INFO] Step 2: calling compiler function debug_wrapper
[2023-10-04 16:31:41,455] torch._inductor.compile_fx: [INFO] Step 3: torchinductor compiling FORWARDS graph 1
[2023-10-04 16:31:41,529] torch._inductor.compile_fx: [INFO] Step 3: torchinductor done compiling FORWARDS graph 1
[2023-10-04 16:31:41,530] torch._dynamo.output_graph: [INFO] Step 2: done compiler function debug_wrapper
[2023-10-04 16:31:41,610] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo start tracing <graph break in pos_encoding>
[2023-10-04 16:31:41,614] torch._dynamo.output_graph: [INFO] Step 2: calling compiler function debug_wrapper
[2023-10-04 16:31:41,624] torch._inductor.compile_fx: [INFO] Step 3: torchinductor compiling FORWARDS graph 2
[2023-10-04 16:31:41,694] torch._inductor.compile_fx: [INFO] Step 3: torchinductor done compiling FORWARDS graph 2
[2023-10-04 16:31:41,694] torch._dynamo.output_graph: [INFO] Step 2: done compiler function debug_wrapper
[2023-10-04 16:31:41,776] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo start tracing <graph break in pos_encoding>
[2023-10-04 16:31:41,779] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo done tracing <graph break in pos_encoding> (RETURN_VALUE)
[2023-10-04 16:31:41,780] torch._dynamo.output_graph: [INFO] Step 2: calling compiler function debug_wrapper
[2023-10-04 16:31:41,789] torch._inductor.compile_fx: [INFO] Step 3: torchinductor compiling FORWARDS graph 3
[2023-10-04 16:31:41,870] torch._inductor.compile_fx: [INFO] Step 3: torchinductor done compiling FORWARDS graph 3
[2023-10-04 16:31:41,870] torch._dynamo.output_graph: [INFO] Step 2: done compiler function debug_wrapper
[2023-10-04 16:31:41,955] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo start tracing <graph break in forward>
[2023-10-04 16:31:42,542] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo done tracing <graph break in forward> (RETURN_VALUE)
[2023-10-04 16:31:42,551] torch._dynamo.output_graph: [INFO] Step 2: calling compiler function debug_wrapper
/home/venom/mambaforge/lib/python3.9/site-packages/torch/autograd/__init__.py:303: UserWarning: Error detected in torch::autograd::GraphRoot. Traceback of forward call that caused the error:
 (Triggered internally at ../torch/csrc/autograd/python_anomaly_mode.cpp:114.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
  0%|                                                                                                                                                                                                                      | 0/2000 [00:03<?, ?it/s]
Traceback (most recent call last):
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/output_graph.py", line 670, in call_user_compiler
    compiled_fn = compiler_fn(gm, self.fake_example_inputs())
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/debug_utils.py", line 1055, in debug_wrapper
    compiled_gm = compiler_fn(gm, example_inputs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/__init__.py", line 1390, in __call__
    return compile_fx(model_, inputs_, config_patches=self.config)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_inductor/compile_fx.py", line 401, in compile_fx
    return compile_fx(
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_inductor/compile_fx.py", line 455, in compile_fx
    return aot_autograd(
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/backends/common.py", line 48, in compiler_fn
    cg = aot_module_simplified(gm, example_inputs, **kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_functorch/aot_autograd.py", line 2822, in aot_module_simplified
    compiled_fn = create_aot_dispatcher_function(
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/utils.py", line 163, in time_wrapper
    r = func(*args, **kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_functorch/aot_autograd.py", line 2515, in create_aot_dispatcher_function
    compiled_fn = compiler_fn(flat_fn, fake_flat_args, aot_config)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_functorch/aot_autograd.py", line 1715, in aot_wrapper_dedupe
    return compiler_fn(flat_fn, leaf_flat_args, aot_config)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_functorch/aot_autograd.py", line 2104, in aot_dispatch_autograd
    fx_g = make_fx(joint_forward_backward, aot_config.decompositions)(
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/fx/experimental/proxy_tensor.py", line 714, in wrapped
    t = dispatch_trace(wrap_key(func, args, fx_tracer), tracer=fx_tracer, concrete_args=tuple(phs))
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/eval_frame.py", line 209, in _fn
    return fn(*args, **kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/fx/experimental/proxy_tensor.py", line 443, in dispatch_trace
    graph = tracer.trace(root, concrete_args)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/eval_frame.py", line 209, in _fn
    return fn(*args, **kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/fx/_symbolic_trace.py", line 778, in trace
    (self.create_arg(fn(*args)),),
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/fx/_symbolic_trace.py", line 652, in flatten_fn
    tree_out = root_fn(*tree_args)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/fx/experimental/proxy_tensor.py", line 459, in wrapped
    out = f(*tensors)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_functorch/aot_autograd.py", line 1158, in traced_joint
    return functionalized_f_helper(primals, tangents)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_functorch/aot_autograd.py", line 1110, in functionalized_f_helper
    f_outs = flat_fn_no_input_mutations(fn, f_primals, f_tangents, meta, keep_input_mutations)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_functorch/aot_autograd.py", line 1078, in flat_fn_no_input_mutations
    outs = flat_fn_with_synthetic_bases_expanded(fn, primals, primals_after_cloning, maybe_tangents, meta, keep_input_mutations)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_functorch/aot_autograd.py", line 1050, in flat_fn_with_synthetic_bases_expanded
    outs = forward_or_joint(fn, primals_before_cloning, primals, maybe_tangents, meta, keep_input_mutations)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_functorch/aot_autograd.py", line 1019, in forward_or_joint
    backward_out = torch.autograd.grad(
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/autograd/__init__.py", line 269, in grad
    return handle_torch_function(
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/overrides.py", line 1534, in handle_torch_function
    result = mode.__torch_function__(public_api, types, args, kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_inductor/overrides.py", line 38, in __torch_function__
    return func(*args, **kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/autograd/__init__.py", line 303, in grad
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/utils/_stats.py", line 20, in wrapper
    return fn(*args, **kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/fx/experimental/proxy_tensor.py", line 487, in __torch_dispatch__
    return self.inner_torch_dispatch(func, types, args, kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/fx/experimental/proxy_tensor.py", line 512, in inner_torch_dispatch
    out = proxy_call(self, func, args, kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/fx/experimental/proxy_tensor.py", line 345, in proxy_call
    out = func(*args, **kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_ops.py", line 287, in __call__
    return self._op(*args, **kwargs or {})
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/utils/_stats.py", line 20, in wrapper
    return fn(*args, **kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_subclasses/fake_tensor.py", line 987, in __torch_dispatch__
    return self.dispatch(func, types, args, kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_subclasses/fake_tensor.py", line 1162, in dispatch
    op_impl_out = op_impl(self, func, *args, **kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_subclasses/fake_tensor.py", line 410, in local_scalar_dense
    raise DataDependentOutputException(func)
torch._subclasses.fake_tensor.DataDependentOutputException: aten._local_scalar_dense.default

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/venom/repo/Diffusion-Models-pytorch/ddpm_conditional.py", line 144, in <module>
    launch()
  File "/home/venom/repo/Diffusion-Models-pytorch/ddpm_conditional.py", line 139, in launch
    train(args)
  File "/home/venom/repo/Diffusion-Models-pytorch/ddpm_conditional.py", line 104, in train
    predicted_noise = traced_model(x_t, t, labels)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/eval_frame.py", line 82, in forward
    return self.dynamo_ctx(self._orig_mod.forward)(*args, **kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/eval_frame.py", line 209, in _fn
    return fn(*args, **kwargs)
  File "/home/venom/repo/Diffusion-Models-pytorch/modules.py", line 229, in forward
    t = self.pos_encoding(t, torch.tensor(self.time_dim).to(t.device))
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/eval_frame.py", line 337, in catch_errors
    return callback(frame, cache_size, hooks)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/convert_frame.py", line 404, in _convert_frame
    result = inner_convert(frame, cache_size, hooks)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/convert_frame.py", line 104, in _fn
    return fn(*args, **kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/convert_frame.py", line 262, in _convert_frame_assert
    return _compile(
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/utils.py", line 163, in time_wrapper
    r = func(*args, **kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/convert_frame.py", line 324, in _compile
    out_code = transform_code_object(code, transform)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/bytecode_transformation.py", line 445, in transform_code_object
    transformations(instructions, code_options)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/convert_frame.py", line 311, in transform
    tracer.run()
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/symbolic_convert.py", line 1726, in run
    super().run()
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/symbolic_convert.py", line 576, in run
    and self.step()
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/symbolic_convert.py", line 540, in step
    getattr(self, inst.opname)(inst)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/symbolic_convert.py", line 1792, in RETURN_VALUE
    self.output.compile_subgraph(
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/output_graph.py", line 517, in compile_subgraph
    self.compile_and_call_fx_graph(tx, list(reversed(stack_values)), root)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/output_graph.py", line 588, in compile_and_call_fx_graph
    compiled_fn = self.call_user_compiler(gm)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/utils.py", line 163, in time_wrapper
    r = func(*args, **kwargs)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_dynamo/output_graph.py", line 675, in call_user_compiler
    raise BackendCompilerFailed(self.compiler_fn, e) from e
torch._dynamo.exc.BackendCompilerFailed: debug_wrapper raised DataDependentOutputException: aten._local_scalar_dense.default

Set torch._dynamo.config.verbose=True for more information


You can suppress this exception and fall back to eager by setting:
    torch._dynamo.config.suppress_errors = True

```