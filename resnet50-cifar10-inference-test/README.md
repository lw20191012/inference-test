这个项目是对比pytorch和onnxruntime部署resnet50模型推理的性能

pytorch推理：

```
python pytorch-resnet50-inference-gpu.py --batch-size 32 --dataset test execution-provider cuda
```

onnxruntime推理

```
python onnxruntime-resnet50-inference-gpu.py --batch-size 32 --dataset test execution-provider cuda
```

