```sh
python ddpm_conditional.py                                                                                                                                                                                                                    ─╯
04:26:53 - INFO: Starting epoch 0:
  0%|                                                                                                                                                                                                                      | 0/2000 [00:00<?, ?it/s][2023-10-04 16:26:53,379] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo start tracing forward
[2023-10-04 16:26:53,387] torch._dynamo.output_graph: [INFO] Step 2: calling compiler function debug_wrapper
[2023-10-04 16:26:55,066] torch._inductor.compile_fx: [INFO] Step 3: torchinductor compiling FORWARDS graph 0
[2023-10-04 16:26:55,154] torch._inductor.compile_fx: [INFO] Step 3: torchinductor done compiling FORWARDS graph 0
[2023-10-04 16:26:55,154] torch._dynamo.output_graph: [INFO] Step 2: done compiler function debug_wrapper
[2023-10-04 16:26:55,297] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo start tracing pos_encoding
[2023-10-04 16:26:55,301] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo start tracing <graph break in pos_encoding>
[2023-10-04 16:26:55,309] torch._dynamo.output_graph: [INFO] Step 2: calling compiler function debug_wrapper
[2023-10-04 16:26:55,330] torch._inductor.compile_fx: [INFO] Step 3: torchinductor compiling FORWARDS graph 1
[2023-10-04 16:26:55,425] torch._inductor.compile_fx: [INFO] Step 3: torchinductor done compiling FORWARDS graph 1
[2023-10-04 16:26:55,426] torch._dynamo.output_graph: [INFO] Step 2: done compiler function debug_wrapper
[2023-10-04 16:26:55,511] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo start tracing <graph break in pos_encoding>
[2023-10-04 16:26:55,515] torch._dynamo.output_graph: [INFO] Step 2: calling compiler function debug_wrapper
[2023-10-04 16:26:55,525] torch._inductor.compile_fx: [INFO] Step 3: torchinductor compiling FORWARDS graph 2
[2023-10-04 16:26:55,577] torch._inductor.compile_fx: [INFO] Step 3: torchinductor done compiling FORWARDS graph 2
[2023-10-04 16:26:55,577] torch._dynamo.output_graph: [INFO] Step 2: done compiler function debug_wrapper
[2023-10-04 16:26:55,658] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo start tracing <graph break in pos_encoding>
[2023-10-04 16:26:55,661] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo done tracing <graph break in pos_encoding> (RETURN_VALUE)
[2023-10-04 16:26:55,661] torch._dynamo.output_graph: [INFO] Step 2: calling compiler function debug_wrapper
[2023-10-04 16:26:55,670] torch._inductor.compile_fx: [INFO] Step 3: torchinductor compiling FORWARDS graph 3
[2023-10-04 16:26:55,739] torch._inductor.compile_fx: [INFO] Step 3: torchinductor done compiling FORWARDS graph 3
[2023-10-04 16:26:55,740] torch._dynamo.output_graph: [INFO] Step 2: done compiler function debug_wrapper
[2023-10-04 16:26:55,819] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo start tracing <graph break in forward>
[2023-10-04 16:26:56,293] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo done tracing <graph break in forward> (RETURN_VALUE)
[2023-10-04 16:26:56,302] torch._dynamo.output_graph: [INFO] Step 2: calling compiler function debug_wrapper
/home/venom/mambaforge/lib/python3.9/site-packages/torch/_inductor/compile_fx.py:90: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
  warnings.warn(
[2023-10-04 16:27:01,274] torch._inductor.compile_fx: [INFO] Step 3: torchinductor compiling FORWARDS graph 4
[2023-10-04 16:27:05,051] torch._inductor.compile_fx: [INFO] Step 3: torchinductor done compiling FORWARDS graph 4
[2023-10-04 16:27:05,052] torch._dynamo.output_graph: [INFO] Step 2: done compiler function debug_wrapper
[2023-10-04 16:27:06,140] torch._inductor.compile_fx: [INFO] Step 3: torchinductor compiling BACKWARDS graph 4
[2023-10-04 16:27:06,745] torch._inductor.graph: [INFO] Using FallbackKernel: aten.max_pool2d_with_indices_backward
[2023-10-04 16:27:10,651] torch._inductor.compile_fx: [INFO] Step 3: torchinductor done compiling BACKWARDS graph 4
  0%|                                                                                                                                                                                                  | 1/2000 [00:17<9:36:45, 17.31s/it, MSE=1.85][2023-10-04 16:27:10,690] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo start tracing <graph break in forward>
[2023-10-04 16:27:11,188] torch._dynamo.symbolic_convert: [INFO] Step 1: torchdynamo done tracing <graph break in forward> (RETURN_VALUE)
[2023-10-04 16:27:11,197] torch._dynamo.output_graph: [INFO] Step 2: calling compiler function debug_wrapper
[2023-10-04 16:27:16,340] torch._inductor.compile_fx: [INFO] Step 3: torchinductor compiling FORWARDS graph 5
[2023-10-04 16:27:19,475] torch._inductor.compile_fx: [INFO] Step 3: torchinductor done compiling FORWARDS graph 5
[2023-10-04 16:27:19,476] torch._dynamo.output_graph: [INFO] Step 2: done compiler function debug_wrapper
[2023-10-04 16:27:19,885] torch._inductor.compile_fx: [INFO] Step 3: torchinductor compiling BACKWARDS graph 5
[2023-10-04 16:27:20,523] torch._inductor.graph: [INFO] Using FallbackKernel: aten.max_pool2d_with_indices_backward
[2023-10-04 16:27:23,574] torch._inductor.compile_fx: [INFO] Step 3: torchinductor done compiling BACKWARDS graph 5
  0%|                                                                                                                                                                                                 | 1/2000 [00:30<16:46:21, 30.21s/it, MSE=1.85]
Traceback (most recent call last):
  File "/home/venom/repo/Diffusion-Models-pytorch/ddpm_conditional.py", line 144, in <module>
    launch()
  File "/home/venom/repo/Diffusion-Models-pytorch/ddpm_conditional.py", line 139, in launch
    train(args)
  File "/home/venom/repo/Diffusion-Models-pytorch/ddpm_conditional.py", line 108, in train
    loss.backward()
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_tensor.py", line 487, in backward
    torch.autograd.backward(
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/autograd/__init__.py", line 200, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/autograd/function.py", line 274, in apply
    return user_fn(self, *args)
  File "/home/venom/mambaforge/lib/python3.9/site-packages/torch/_functorch/aot_autograd.py", line 2327, in backward
    list(ctx.symints) + list(ctx.saved_tensors) + list(contiguous_args)
RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.

```