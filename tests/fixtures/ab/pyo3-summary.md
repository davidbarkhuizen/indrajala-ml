
### Array methods (boundary_probe.py)

| case | old (ns) | new (ns) | new/old |
|---|---|---|---|
| shape (getter, tuple out) | 128.9-169.4 | 112.4-124.2 | 0.76 |
| a[i, j] (index in, float out) | 203.7-254.9 | 148.7-159.7 | 0.66 |
| a[:, :-1] (slices in, Array out) | 358.2-494.2 | 323.9-343.4 | 0.86 |
| a[i, j] = x | 206.3-264.3 | 172.4-191.8 | 0.81 |
| a + b (Array operand) | 315.6-418.4 | 276-301.9 | 0.83 |
| a + v (broadcast) | 705.4-850.5 | 665.5-811.8 | 0.92 |
| a * 0.5 (scalar operand) | 383.1-407.6 | 293-325.3 | 0.78 |
| 0.5 * a (reflected) | 435.4-485.3 | 328.5-362.7 | 0.75 |
| a += b (in place) | 290.4-331.6 | 259.7-282.4 | 0.85 |
| a.tolist() 10x30 | 5371-6990 | 3801-4199 | 0.71 |
| v.tolist() 30 | 511.3-597.8 | 356.5-422.8 | 0.71 |
| Array(nested 10x30) | 5034-5743 | 3693-4586 | 0.71 |
| Array.from_rows(10x30) | 3206-4015 | 1918-2065 | 0.59 |
| Array.zeros((10, 30)) | 364.2-459.7 | 324.3-397 | 0.76 |
| layer_forward 10x30 (&RustArray args) | 1268-1348 | 1113-1223 | 0.89 |
| seed(0) (int seed) | 2454-2763 | 2319-2734 | 0.90 |
| seed([1, 2, 3]) (sequence seed) | 9563-1.034e+04 | 9306-1.013e+04 | 0.96 |

### Fused dense ops (focused_benchmark.py)

| case | old (µs) | new (µs) | new/old |
|---|---|---|---|
| dense 30 x 784 forward | 3.781-4.501 | 3.567-4.257 | 0.95 |
| dense 30 x 784 downstream | 3.36-4.083 | 3.261-3.94 | 0.92 |
| dense 30 x 784 hidden_delta | 3.816-4.796 | 4.109-4.43 | 0.97 |
| dense 30 x 784 accumulate_gradient | 8.429-10.99 | 6.799-8.65 | 0.91 |
| dense 30 x 784 apply_accumulated_gradient | 20.89-24.71 | 18.38-23.86 | 1.00 |
| dense 30 x 784 sgd step | 8.282-10.18 | 8.112-10.21 | 0.96 |
| dense 30 x 784 forward_batch b1 | 3.772-5.06 | 3.75-4.434 | 0.97 |
| dense 30 x 784 downstream_batch b1 | 3.262-4.238 | 3.244-4.389 | 0.96 |
| dense 30 x 784 hidden_delta_batch b1 | 4.17-4.85 | 3.896-4.467 | 0.90 |
| dense 30 x 784 accumulate_gradient_batch b1 | 11.91-18.46 | 11.98-14 | 0.96 |
| dense 10 x 30 forward | 0.7878-0.8789 | 0.7177-0.8156 | 0.98 |
| dense 10 x 30 downstream | 0.489-0.5675 | 0.399-0.54 | 0.81 |
| dense 10 x 30 hidden_delta | 0.6269-0.7792 | 0.5979-0.7418 | 0.92 |
| dense 10 x 30 accumulate_gradient | 0.9854-1.249 | 0.8547-1.042 | 0.87 |
| dense 10 x 30 apply_accumulated_gradient | 1.948-2.673 | 1.817-2.342 | 0.85 |
| dense 10 x 30 sgd step | 0.9457-1.167 | 0.8747-1.047 | 0.82 |
| dense 10 x 30 forward_batch b1 | 0.7842-0.942 | 0.7094-0.8792 | 0.89 |
| dense 10 x 30 downstream_batch b1 | 0.444-0.5815 | 0.4548-0.5062 | 0.89 |
| dense 10 x 30 hidden_delta_batch b1 | 0.6685-0.7861 | 0.6168-0.6834 | 0.86 |
| dense 10 x 30 accumulate_gradient_batch b1 | 1.212-1.305 | 1.125-1.369 | 1.01 |
| dense 30 x 784 bare downstream b1 | 3.194-3.749 | 3.186-3.768 | 0.98 |
| dense 30 x 784 bare accumulate b1 | 8.339-9.825 | 8.639-9.958 | 0.99 |
| dense 30 x 784 transpose b1 | 0.2778-0.3657 | 0.2838-0.3517 | 0.89 |
| dense 30 x 784 add b1 | 6.145-9.495 | 6.046-8.264 | 0.97 |
| dense 30 x 784 sum_axis0 b1 | 0.3065-0.3758 | 0.2412-0.3266 | 0.85 |
| dense 10 x 30 bare downstream b1 | 0.3299-0.4147 | 0.306-0.3614 | 0.80 |
| dense 10 x 30 bare accumulate b1 | 0.5605-0.6477 | 0.5269-0.5882 | 0.88 |
| dense 10 x 30 transpose b1 | 0.2636-0.2805 | 0.2103-0.274 | 0.95 |
| dense 10 x 30 add b1 | 0.3677-0.4343 | 0.3315-0.3686 | 0.83 |
| dense 10 x 30 sum_axis0 b1 | 0.3014-0.35 | 0.2493-0.3531 | 0.97 |

### Conv epoch, seconds per op (epoch_op_profile.py)

| case | old (s) | new (s) | new/old |
|---|---|---|---|
| conv / single-example (profiled total) | 0.6873-1.033 | 0.7427-0.9788 | 1.05 |
| conv / single-example layer_sgd_step | 0.1501-0.2661 | 0.1525-0.2341 | 1.08 |
| conv / single-example layer_forward | 0.1413-0.2348 | 0.1595-0.2054 | 1.06 |
| conv / single-example conv_forward_batch | 0.1563-0.201 | 0.168-0.2153 | 1.09 |
| conv / single-example layer_downstream | 0.05024-0.07323 | 0.05472-0.07097 | 1.08 |
| conv / single-example conv_accumulate_gradient_batch | 0.05075-0.06294 | 0.05529-0.06757 | 1.05 |
| conv / single-example Array.row | 0.006557-0.008869 | 0.006472-0.008978 | 0.97 |
| conv / single-example array_relu_mask | 0.005805-0.02966 | 0.005428-0.007606 | 1.01 |
| conv / mini-batch (32) (profiled total) | 0.5564-0.7808 | 0.6033-0.7337 | 1.09 |
| conv / mini-batch (32) conv_forward_batch | 0.152-0.2255 | 0.1701-0.2265 | 1.08 |
| conv / mini-batch (32) layer_forward | 0.08183-0.1186 | 0.09394-0.1293 | 1.04 |
| conv / mini-batch (32) conv_accumulate_gradient_batch | 0.06653-0.0892 | 0.06748-0.08179 | 1.08 |
| conv / mini-batch (32) layer_accumulate_gradient_batch | 0.05546-0.07653 | 0.05653-0.126 | 1.12 |
| conv / mini-batch (32) layer_forward_batch | 0.04567-0.05973 | 0.04774-0.05685 | 1.06 |
| conv / mini-batch (32) layer_downstream_batch | 0.03929-0.05405 | 0.0417-0.05188 | 1.08 |
| conv / mini-batch (32) array_relu_mask | 0.01807-0.0226 | 0.01988-0.02306 | 1.03 |
| conv / mini-batch (32) layer_apply_accumulated_gradient | 0.01235-0.01971 | 0.01324-0.02005 | 1.09 |
| conv / mini-batch (32) Array.row | 0.005271-0.005271 | 0.005179-0.005179 | 0.98 |
