import onnxruntime
import numpy as np
import time
import matplotlib.pyplot as plt
import torch

onnx_model_path = "yolov5s.onnx"

# 创建ONNX Runtime会话
options_unoptimized = onnxruntime.SessionOptions()
options_unoptimized.enable_profiling = True
options_unoptimized.enable_mem_reuse = False
options_unoptimized.enable_mem_pattern = False
options_unoptimized.enable_cpu_mem_arena = False
options_unoptimized.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

ort_session_optimized = onnxruntime.InferenceSession(onnx_model_path)
ort_session_unoptimized = onnxruntime.InferenceSession(onnx_model_path, sess_options=options_unoptimized)

# 准备适当的输入数据，这里使用随机数据作为示例
input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)

# 进行五次推理并记录时间
num_runs = 5
optimized_times = []
unoptimized_times = []

for _ in range(num_runs):
    # 开启优化的推理
    ort_inputs_optimized = {ort_session_optimized.get_inputs()[0].name: input_data}
    torch.cuda.synchronize()
    start_time_optimized = time.time()
    outputs_optimized = ort_session_optimized.run(None, ort_inputs_optimized)
    torch.cuda.synchronize()
    end_time_optimized = time.time()
    optimized_times.append(end_time_optimized - start_time_optimized)

    # 关闭优化的推理
    ort_inputs_unoptimized = {ort_session_unoptimized.get_inputs()[0].name: input_data}
    torch.cuda.synchronize()
    start_time_unoptimized = time.time()
    outputs_unoptimized = ort_session_unoptimized.run(None, ort_inputs_unoptimized)
    torch.cuda.synchronize()
    end_time_unoptimized = time.time()
    unoptimized_times.append(end_time_unoptimized - start_time_unoptimized)

# 打印和绘制推理时间曲线图
print("Optimized Inference Times:", optimized_times)
print("Unoptimized Inference Times:", unoptimized_times)

plt.plot(range(1, num_runs + 1), optimized_times, label='Optimized')
plt.plot(range(1, num_runs + 1), unoptimized_times, label='Unoptimized')
plt.xlabel('Run')
plt.ylabel('Inference Time (seconds)')
plt.title(f'Inference Time Comparison for {num_runs} Runs')
plt.legend()
plt.show()
