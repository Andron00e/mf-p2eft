from functools import partial
import time
import pathlib

import torch
from torch.utils.cpp_extension import load

tropical_cuda_path = str(pathlib.Path(__file__).parent.resolve() / "tropical.cu")
tropical_cuda = load(name="tropical", sources=[tropical_cuda_path], verbose=True)


def reference_mp(X, W):
    sum_var = W.view(1, W.shape[0], W.shape[1]) + X.view(X.shape[0], 1, X.shape[1])
    return sum_var.max(axis=2).values


def reference_mmpp(X, W):
    sum_var = W.view(1, W.shape[0], W.shape[1]) + X.view(X.shape[0], 1, X.shape[1])
    return sum_var.max(axis=2).values + sum_var.min(axis=2).values


def track_multi_activation(W_activation, K_extremum, label):
    if W_activation is None:
        return
    for i in range(K_extremum.shape[0]):
        for j in range(K_extremum.shape[2]):
            W_activation.scatter_(
                1,
                K_extremum[i, :, j]
                .view(-1, 1)
                .type(torch.int64)
                .to(W_activation.device),
                1,
                reduce="add",
            )


def track_activation(W_activation, K_extremum, label):
    if W_activation is None:
        return
    for i in range(K_extremum.shape[0]):
        W_activation.scatter_(
            1,
            K_extremum[i, :].view(-1, 1).type(torch.int64).to(W_activation.device),
            1,
            reduce="add",
        )


class tropical_minp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, W_activation, normalized_W):
        assert X.shape[1] == W.shape[1], f"{X.shape=} {W.shape=}"
        Y, K = tropical_cuda.minp_forward(X.contiguous(), W)
        ctx.save_for_backward(
            X, W, K
        )  # Only need the shapes and device for X, W
        track_activation(W_activation, K, "min")
        if normalized_W is not None:
            max_updates, _ = tropical_cuda.mp_forward(
                -X.T.contiguous(), Y.T.contiguous()
            )
            normalized_W = normalized_W.to(max_updates.device)
            if torch.sum(normalized_W) != 0:
                normalized_W = torch.max(max_updates.T, normalized_W)
            else:
                normalized_W += max_updates.T
        return Y

    @staticmethod
    def backward(ctx, Y_grad):
        X, W, K = ctx.saved_tensors
        W_grad = torch.empty_like(W)
        X_grad = torch.empty_like(X)
        tropical_cuda.minp_backward_x(Y_grad, X_grad, K)
        tropical_cuda.minp_backward_x(Y_grad.T.contiguous(), W_grad, K.T.contiguous())
        return X_grad, W_grad, None, None


class tropical_maxp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, W_activation, normalized_W):
        assert X.shape[1] == W.shape[1], f"{X.shape=} {W.shape=}"
        Y, K = tropical_cuda.mp_forward(X, W)
        ctx.save_for_backward(
            X, W, K
        )  # Only need the shapes and device for X, W
        track_activation(W_activation, K, "max")
        if normalized_W is not None:
            min_updates, _ = tropical_cuda.minp_forward(
                -X.T.contiguous(), Y.T.contiguous()
            )
            normalized_W = normalized_W.to(min_updates.device)
            if torch.sum(normalized_W) != 0:
                normalized_W = torch.min(min_updates.T, normalized_W)
            else:
                normalized_W += min_updates.T
        return Y

    @staticmethod
    def backward(ctx, Y_grad):
        X, W, K = ctx.saved_tensors
        X_grad = torch.empty_like(X)
        W_grad = torch.empty_like(W)
        tropical_cuda.mp_backward_x(Y_grad, X_grad, K)
        tropical_cuda.mp_backward_x(Y_grad.T.contiguous(), W_grad, K.T.contiguous())
        return X_grad, W_grad, None, None


class tropical_multiminp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, W_activation):
        assert X.shape[1] == W.shape[1], f"{X.shape=} {W.shape=}"
        Y, K = tropical_cuda.multi_minp_forward(
            X,
            W,
        )
        ctx.save_for_backward(X, W, K)
        track_multi_activation(W_activation, K, "min")
        return Y

    @staticmethod
    def backward(ctx, Y_grad):
        X, W, K = ctx.saved_tensors
        X_grad = torch.empty_like(X)
        W_grad = torch.empty_like(W)
        tropical_cuda.multi_minp_backward_x(Y_grad, X_grad, K)
        tropical_cuda.multi_minp_backward_x(
            Y_grad.T.contiguous(), W_grad, torch.transpose(K, 0, 1).contiguous()
        )
        return X_grad, W_grad, None


class tropical_multimax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, W_activation):
        assert X.shape[1] == W.shape[1], f"{X.shape=} {W.shape=}"
        Y, K = tropical_cuda.multi_mp_forward(
            X,
            W,
        )
        ctx.save_for_backward(X, W, K)
        track_multi_activation(W_activation, K, "max")
        return Y

    @staticmethod
    def backward(ctx, Y_grad):
        X, W, K = ctx.saved_tensors
        X_grad = torch.empty_like(X)
        W_grad = torch.empty_like(W)
        tropical_cuda.multi_mp_backward_x(Y_grad, X_grad, K)
        tropical_cuda.multi_mp_backward_x(
            Y_grad.T.contiguous(), W_grad, torch.transpose(K, 0, 1).contiguous()
        )
        return X_grad, W_grad, None


class solidmaxmin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W_MAX, W_MIN, W_activation_MAX, W_activation_MIN):
        Y_MIN, K_MIN = tropical_cuda.minp_forward(X.contiguous(), W_MIN)
        Y_MAX, K_MAX = tropical_cuda.mp_forward(X, W_MAX)
        track_activation(W_activation_MAX, K_MAX, "max")
        track_activation(W_activation_MIN, K_MIN, "min")
        ctx.save_for_backward(
            X, W_MIN, K_MIN, W_MAX, K_MAX
        )  # Only need the shapes and device for X, W
        return Y_MIN + Y_MAX

    @staticmethod
    def backward(ctx, Y_grad):
        X, W_MIN, K_MIN, W_MAX, K_MAX = ctx.saved_tensors
        N = X.shape[1]
        X_grad = torch.empty_like(X)
        W_grad_MIN = torch.empty_like(W_MIN)
        W_grad_MAX = torch.empty_like(W_MAX)
        tropical_cuda.minp_backward_x(Y_grad, X_grad, K_MIN)
        tropical_cuda.minp_backward_x(
            Y_grad.T.contiguous(), W_grad_MIN, K_MIN.T.contiguous()
        )
        tropical_cuda.mp_backward_x(Y_grad, X_grad, K_MAX)
        tropical_cuda.mp_backward_x(
            Y_grad.T.contiguous(), W_grad_MAX, K_MAX.T.contiguous()
        )
        return X_grad, W_grad_MAX, W_grad_MIN, None, None


class leaky_tropical_minp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, W_activation, leak_factor, is_random):
        assert X.shape[1] == W.shape[1], f"{X.shape=} {W.shape=}"
        Y, K = tropical_cuda.minp_forward(X.contiguous(), W)
        ctx.save_for_backward(
            X, W, K, leak_factor, is_random
        )  # Only need the shapes and device for X, W
        track_activation(W_activation, K, "leaky min")
        return Y

    @staticmethod
    def backward(ctx, Y_grad):
        X, W, K, leak_factor, is_random = ctx.saved_tensors
        X_grad = torch.zeros_like(X)
        W_grad = torch.zeros_like(W)
        tropical_cuda.minp_backward_x(Y_grad, X_grad, K)
        tropical_cuda.minp_backward_x(Y_grad.T.contiguous(), W_grad, K.T.contiguous())
        avg_update = torch.abs(torch.mean(Y_grad))
        if is_random > 0:
            X_grad += torch.normal(
                0, leak_factor[0] * avg_update, size=X.shape, device=X.device
            )
            W_grad += torch.normal(
                0, leak_factor[0] * avg_update, size=W.shape, device=W.device
            )
        else:
            X_grad += leak_factor[0] * torch.ones_like(X)
            W_grad += leak_factor[0] * torch.ones_like(W)
        return X_grad, W_grad, None, None, None


class leaky_tropical_maxp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, W_activation, leak_factor, is_random):
        assert X.shape[1] == W.shape[1], f"{X.shape=} {W.shape=}"
        (
            Y,
            K,
        ) = tropical_cuda.mp_forward(X, W)
        ctx.save_for_backward(
            X, W, K, leak_factor, is_random
        )  # Only need the shapes and device for X, W
        track_activation(W_activation, K, "leaky max")
        return Y

    @staticmethod
    def backward(ctx, Y_grad):
        X, W, K, leak_factor, is_random = ctx.saved_tensors
        X_grad = torch.zeros_like(X)
        W_grad = torch.zeros_like(W)
        tropical_cuda.mp_backward_x(Y_grad, X_grad, K)
        tropical_cuda.mp_backward_x(Y_grad.T.contiguous(), W_grad, K.T.contiguous())
        avg_update = torch.abs(torch.mean(Y_grad))
        if is_random > 0:
            X_grad += torch.normal(
                0, leak_factor[0] * avg_update, size=X.shape, device=X.device
            )
            W_grad += torch.normal(
                0, leak_factor[0] * avg_update, size=W.shape, device=W.device
            )
        else:
            X_grad += leak_factor[0] * torch.ones_like(X)
            W_grad += leak_factor[0] * torch.ones_like(W)
        return X_grad, W_grad, None, None, None


class tropical_mmpp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W):
        assert X.shape[1] == W.shape[1], f"{X.shape=} {W.shape=}"
        Y, K_max, K_min = tropical_cuda.mmpp_forward(X, W)
        ctx.save_for_backward(
            X, W, K_max, K_min
        )  # Only need the shapes and device for X, W
        return Y

    @staticmethod
    def backward(ctx, Y_grad):
        X, W, K_max, K_min = ctx.saved_tensors
        X_grad = torch.empty_like(X)
        W_grad = torch.empty_like(W)
        tropical_cuda.mmpp_backward_x(Y_grad, X_grad, K_max, K_min)
        tropical_cuda.mmpp_backward_x(
            Y_grad.T.contiguous(), W_grad, K_max.T.contiguous(), K_min.T.contiguous()
        )
        return X_grad, W_grad


def folded_conv(linear_function, X, W, stride=1, padding=0, dilation=1):
    # https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
    # https://github.com/pytorch/pytorch/issues/47990
    N, _, H_in, W_in = X.shape
    C_out, _, kH, kW = W.shape
    H_out = (H_in + 2 * padding - (kH - 1) - 1) // stride + 1
    W_out = (W_in + 2 * padding - (kW - 1) - 1) // stride + 1

    # N, stacked_channels, spatial_locations
    X_unfolded = torch.nn.functional.unfold(
        X, (kH, kW), padding=padding, stride=stride, dilation=dilation
    )
    _, stacked_channels, spatial_locations = X_unfolded.shape

    # N, spatial_locations, stacked_channels
    X_unfolded_T = X_unfolded.transpose(1, 2).contiguous()

    Y_unfolded = (
        linear_function(
            X_unfolded_T.view(-1, stacked_channels),
            W.view(W.size(0), -1),
        )
        .view(N, spatial_locations, C_out)
        .transpose(1, 2)
    )
    Y = Y_unfolded.view(N, C_out, H_out, W_out)
    return Y


def tropical_mmpp_conv(X, W, stride=1, padding=0, dilation=1):
    return folded_conv(tropical_mmpp.apply, X, W, stride, padding, dilation)


def test_linear(function, reference_function, N=1015, D_in=1016, D_out=1017):
    X1 = torch.randn((N, D_in), device="cuda", requires_grad=True)
    W1 = torch.randn((D_out, D_in), device="cuda", requires_grad=True)

    X2 = X1.clone().detach().requires_grad_(True)
    W2 = W1.clone().detach().requires_grad_(True)

    Y_grad = torch.randn((N, D_out), device="cuda").round(decimals=2)

    Y1 = function(X1, W1)
    Y1.backward(Y_grad)

    Y2 = reference_function(X2, W2)
    Y2.backward(Y_grad)

    assert torch.allclose(Y1, Y2, atol=1e-3, rtol=1e-4)
    assert torch.allclose(W1.grad.round(decimals=4), W2.grad.round(decimals=4))
    assert torch.allclose(X1.grad.round(decimals=4), X2.grad.round(decimals=4))


def test_linear_speed(methods, N=128, D_in=1024, D_out=1024, iters=1000, num_mats=100):
    # Note that torch matmul is more efficient for larger N
    Xs = [
        torch.randn((N, D_in), device="cuda", requires_grad=True)
        for _ in range(num_mats)
    ]
    Ws = [
        torch.randn((D_out, D_in), device="cuda", requires_grad=True)
        for _ in range(num_mats)
    ]
    Y_grads = [torch.randn((N, D_out), device="cuda") for _ in range(num_mats)]
    return test_speed(methods, Xs, Ws, Y_grads, iters)


def test_conv_speed(
    methods,
    X_shape=(128, 128, 31, 31),
    W_shape=(128, 128, 1, 1),
    padding=0,
    iters=1000,
    num_mats=10,
):
    methods = {key: partial(method, padding=padding) for key, method in methods.items()}
    Xs = [
        torch.randn(X_shape, device="cuda", requires_grad=True) for _ in range(num_mats)
    ]
    Ws = [
        torch.randn(W_shape, device="cuda", requires_grad=True) for _ in range(num_mats)
    ]
    Y_example = torch.nn.functional.conv2d(Xs[0], Ws[0], padding=padding)
    Y_grads = [torch.randn_like(Y_example, device="cuda") for _ in range(num_mats)]
    return test_speed(methods, Xs, Ws, Y_grads, iters)


def test_speed(methods, Xs, Ws, Y_grads, iters):
    num_mats = len(Xs)
    speeds = {}
    for method_name, method_func in methods.items():
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for idx in range(iters):
                Y = method_func(Xs[idx % num_mats], Ws[idx % num_mats])
        torch.cuda.synchronize()
        fwd_dt = time.perf_counter() - start
        speeds[method_name + "_fwd"] = fwd_dt / iters

        Y = method_func(Xs[0], Ws[0])
        torch.cuda.synchronize()
        start = time.perf_counter()
        for idx in range(iters):
            Y.backward(Y_grads[idx % num_mats], retain_graph=True)
        torch.cuda.synchronize()
        bwd_dt = time.perf_counter() - start
        speeds[method_name + "_bwd"] = bwd_dt / iters
        del Y

        torch.cuda.synchronize()
        start = time.perf_counter()
        for idx in range(iters):
            Y = method_func(Xs[idx % num_mats], Ws[idx % num_mats])
            Y.backward(Y_grads[idx % num_mats])
        torch.cuda.synchronize()
        dt = time.perf_counter() - start
        speeds[method_name + "_all"] = dt / iters
        del Y

    return speeds


def test_folded_conv(
    N=128, C_in=32, H=15, W=15, C_out=64, kH=3, kW=3, stride=2, padding=1, dilation=1
):
    function = folded_conv
    reference_function = torch.nn.functional.conv2d
    conv_kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    X1 = torch.randint(
        -1, 2, (N, C_in, H, W), device="cuda", requires_grad=True, dtype=torch.float32
    )
    W1 = torch.randint(
        -1,
        2,
        (C_out, C_in, kH, kW),
        device="cuda",
        requires_grad=True,
        dtype=torch.float32,
    )

    X2 = X1.clone().detach().requires_grad_(True)
    W2 = W1.clone().detach().requires_grad_(True)

    Y1 = function(torch.nn.functional.linear, X1, W1, **conv_kwargs)
    Y2 = reference_function(X2, W2, **conv_kwargs)

    Y_grad = torch.randint_like(Y1, -1, 2)

    Y1.backward(Y_grad)
    Y2.backward(Y_grad)

    assert torch.allclose(Y1, Y2)
    assert torch.allclose(W1.grad, W2.grad)
    assert torch.allclose(X1.grad, X2.grad)


if __name__ == "__main__":
    print("===== Tropical MP =====")
    try:
        test_linear(tropical_mp.apply, reference_mp)
        print("Output test passed")
    except Exception as e:
        print(f"Output test failed {e}")
    methods = {
        "kernel": tropical_mp.apply,
        "native": reference_mp,
        "linear": torch.nn.functional.linear,
    }
    timing = test_linear_speed(methods, N=1024, iters=1000)
    for method_name in methods:
        for direction in ("all", "fwd", "bwd"):
            dt = timing[method_name + "_" + direction]
            ref_dt = timing["kernel" + "_" + direction]
            print(f"{method_name} ({direction}): {dt:0.3e} ({dt/ref_dt:0.3f}x)")

    print("\n===== Tropical MMPP =====")
    try:
        test_linear(tropical_mmpp.apply, reference_mmpp)
        print("Output test passed")
    except Exception as e:
        print(f"Output test failed {e}")
    methods = {
        "kernel": tropical_mmpp.apply,
        "native": reference_mmpp,
        "linear": torch.nn.functional.linear,
    }
    timing = test_linear_speed(methods, N=1024, iters=1000)
    for method_name in methods:
        for direction in ("all", "fwd", "bwd"):
            dt = timing[method_name + "_" + direction]
            ref_dt = timing["kernel" + "_" + direction]
            print(f"{method_name} ({direction}): {dt:0.3e} ({dt/ref_dt:0.3f}x)")

    print("\n===== Folded conv =====")
    try:
        test_folded_conv()
        print("Output test passed")
    except Exception:
        print("Output test failed")

    print("\n===== Tropical MMPP Conv =====")
    methods = {
        "kernel": tropical_mmpp_conv,
        "linear": torch.nn.functional.conv2d,
    }
    timing = test_conv_speed(methods)
    for method_name in methods:
        for direction in ("all", "fwd", "bwd"):
            dt = timing[method_name + "_" + direction]
            ref_dt = timing["kernel" + "_" + direction]
            print(f"{method_name} ({direction}): {dt:0.3e} ({dt/ref_dt:0.3f}x)")
