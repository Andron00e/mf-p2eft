#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

///////////////////////////////////////////////////////////////////////////////////////////////////
// MultiMaxPlus
///////////////////////////////////////////////////////////////////////////////////////////////////
#define BLOCK_SIZE 16
template <typename scalar_t>
__global__ void cuda_multi_mp_forward(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> W,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y,
    torch::PackedTensorAccessor32<int32_t,3,torch::RestrictPtrTraits> K
) {
    // Compute MaxPlus version of X @ W.T
    // X is (BATCH, D_in)
    // W is (D_out, D_in)
    // Y, K are (BATCH, D_out)
    // Y are the results and K are the max indices for bwd pass

    // Shared memory tiles (DO WE NEED AN OFFSET FOR INNER???)
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Output row and column
    const int o_r = blockIdx.y * blockDim.y + threadIdx.y;
    const int o_c = blockIdx.x * blockDim.x + threadIdx.x;

    // Block row and column
    const int r = threadIdx.y;
    const int c = threadIdx.x;

    bool in_bounds = (o_r < Y.size(0)) && (o_c < Y.size(1));

    float y_max_1 = -INFINITY;
    float y_max_2 = -INFINITY;
    float y_max_3 = -INFINITY;
    int k_1 = -1;
    int k_2 = -1;
    int k_3 = -1;
    float sum;
    for(int i=0; i<(X.size(1) + BLOCK_SIZE - 1)/BLOCK_SIZE; ++i){
        // Load sub matrix A
        if (o_r < X.size(0) && i * BLOCK_SIZE + c < X.size(1)){
            As[r][c] = X[o_r][i * BLOCK_SIZE + c]; 
        } else {
            As[r][c] = 0;
        }

        // Load sub matrix B
        if (o_c < W.size(0) && i * BLOCK_SIZE + r < W.size(1)){
            Bs[r][c] = W[o_c][i * BLOCK_SIZE + r]; // Note W transposed, strided access problem ???
        } else {
            Bs[r][c] = 0;
        }

        __syncthreads(); // Finish load before compute
        for(int j=0; j<BLOCK_SIZE && i * BLOCK_SIZE + j < X.size(1); ++j){
            // Note that for r or c out of bounds we are still computing an (incorrect) value
            sum = As[r][j] + Bs[j][c];
            if(y_max_1 < sum){
                y_max_3 = y_max_2;
                k_3 = k_2;

                y_max_2 = y_max_1;
                k_2 = k_1;

                y_max_1 = sum;
                k_1 = i * BLOCK_SIZE + j;
            } else if(y_max_2 < sum){
                y_max_3 = y_max_2;
                k_3 = k_2;

                y_max_2 = sum;
                k_2 = i * BLOCK_SIZE + j;
            } else if(y_max_3 < sum){
                y_max_3 = sum;
                k_3 = i * BLOCK_SIZE + j;
            }


        }
        __syncthreads(); // Finish compute before next load
    }

    if (in_bounds){
        Y[o_r][o_c] = y_max_1 + y_max_2 + y_max_3;
        K[o_r][o_c][0] = k_1;
        K[o_r][o_c][1] = k_2;
        K[o_r][o_c][2] = k_3;
    }
}


std::vector<torch::Tensor> multi_mp_forward(
    torch::Tensor X,
    torch::Tensor W
){
    CHECK_INPUT(X);
    CHECK_INPUT(W);

    assert(X.size(1) == W.size(1));

    auto Y_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto Y = torch::zeros({X.size(0), W.size(0)}, Y_options);

    auto K_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto K = torch::zeros({X.size(0), W.size(0), 3}, K_options);

    const int block_width = BLOCK_SIZE;
    const dim3 block_dim(block_width, block_width); // blockDim.x, blockDim.y
    const dim3 grid_dim((Y.size(1) + block_width - 1) / block_width, (Y.size(0) + block_width - 1) / block_width);

    AT_DISPATCH_FLOATING_TYPES(X.type(), "cuda_multi_mp_forward", ([&] {
        cuda_multi_mp_forward<scalar_t><<<grid_dim, block_dim>>>(
            X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            W.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            Y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            K.packed_accessor32<int32_t,3,torch::RestrictPtrTraits>()
        );
    }));

    return {Y, K};
}


template <typename scalar_t>
__global__ void cuda_multi_mp_backward_x(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_grad,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_grad,
    const torch::PackedTensorAccessor32<int32_t,3,torch::RestrictPtrTraits> K
) {
    const int o_r = blockIdx.y * blockDim.y + threadIdx.y;
    const int o_c = blockIdx.x * blockDim.x + threadIdx.x;

    if (o_r >= X_grad.size(0) || o_c >= X_grad.size(1)){
        return;
    }

    float out = 0.0;
    for(int dim = 0; dim < 3; ++dim) {
        for(int i=0; i<Y_grad.size(1); ++i){
            if(K[o_r][i][dim] == o_c){
                out += Y_grad[o_r][i];
            }
        }
    }
    X_grad[o_r][o_c] = out;
}


void multi_mp_backward_x(
    torch::Tensor Y_grad,
    torch::Tensor X_grad, // Allocate in python
    torch::Tensor K
){
    // Can use this for the bwd pass of W as well, transpose Y_grad and K
    CHECK_INPUT(Y_grad);
    CHECK_INPUT(X_grad);
    CHECK_INPUT(K);

    assert(Y_grad.size(0) == K.size(0));
    assert(Y_grad.size(1) == K.size(1));

    const dim3 block_dim(128, 4); // blockDim.x, blockDim.y
    const dim3 grid_dim(
        (X_grad.size(1) + block_dim.x - 1) / block_dim.x,
        (X_grad.size(0) + block_dim.y - 1) / block_dim.y
    );

    AT_DISPATCH_FLOATING_TYPES(X_grad.type(), "cuda_multi_mp_backward_x", ([&] {
        cuda_multi_mp_backward_x<scalar_t><<<grid_dim, block_dim>>>(
            Y_grad.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            X_grad.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            K.packed_accessor32<int32_t,3,torch::RestrictPtrTraits>()
        );
    }));
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// MultiMinPlus
///////////////////////////////////////////////////////////////////////////////////////////////////
#define BLOCK_SIZE 16
template <typename scalar_t>
__global__ void cuda_multi_minp_forward(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> W,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y,
    torch::PackedTensorAccessor32<int32_t,3,torch::RestrictPtrTraits> K
) {
    // Compute MaxPlus version of X @ W.T
    // X is (BATCH, D_in)
    // W is (D_out, D_in)
    // Y, K are (BATCH, D_out)
    // Y are the results and K are the max indices for bwd pass

    // Shared memory tiles (DO WE NEED AN OFFSET FOR INNER???)
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Output row and column
    const int o_r = blockIdx.y * blockDim.y + threadIdx.y;
    const int o_c = blockIdx.x * blockDim.x + threadIdx.x;

    // Block row and column
    const int r = threadIdx.y;
    const int c = threadIdx.x;

    bool in_bounds = (o_r < Y.size(0)) && (o_c < Y.size(1));

    float y_min_1 = INFINITY;
    float y_min_2 = INFINITY;
    float y_min_3 = INFINITY;
    int k_1 = -1;
    int k_2 = -1;
    int k_3 = -1;
    float sum;
    for(int i=0; i<(X.size(1) + BLOCK_SIZE - 1)/BLOCK_SIZE; ++i){
        // Load sub matrix A
        if (o_r < X.size(0) && i * BLOCK_SIZE + c < X.size(1)){
            As[r][c] = X[o_r][i * BLOCK_SIZE + c]; 
        } else {
            As[r][c] = 0;
        }

        // Load sub matrix B
        if (o_c < W.size(0) && i * BLOCK_SIZE + r < W.size(1)){
            Bs[r][c] = W[o_c][i * BLOCK_SIZE + r]; // Note W transposed, strided access problem ???
        } else {
            Bs[r][c] = 0;
        }

        __syncthreads(); // Finish load before compute
        for(int j=0; j<BLOCK_SIZE && i * BLOCK_SIZE + j < X.size(1); ++j){
            // Note that for r or c out of bounds we are still computing an (incorrect) value
            sum = As[r][j] + Bs[j][c];
            if(y_min_1 > sum){
                y_min_3 = y_min_2;
                k_3 = k_2;

                y_min_2 = y_min_1;
                k_2 = k_1;

                y_min_1 = sum;
                k_1 = i * BLOCK_SIZE + j;
            } else if(y_min_2 > sum){
                y_min_3 = y_min_2;
                k_3 = k_2;

                y_min_2 = sum;
                k_2 = i * BLOCK_SIZE + j;
            } else if(y_min_3 > sum){
                y_min_3 = sum;
                k_3 = i * BLOCK_SIZE + j;
            }
        }
        __syncthreads(); // Finish compute before next load
    }

    if (in_bounds){
        Y[o_r][o_c] = y_min_1 + y_min_2 + y_min_3;
        K[o_r][o_c][0] = k_1;
        K[o_r][o_c][1] = k_2;
        K[o_r][o_c][2] = k_3;
    }
}

std::vector<torch::Tensor> multi_minp_forward(
    torch::Tensor X,
    torch::Tensor W
){
    CHECK_INPUT(X);
    CHECK_INPUT(W);

    assert(X.size(1) == W.size(1));

    auto Y_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto Y = torch::zeros({X.size(0), W.size(0)}, Y_options);

    auto K_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto K = torch::zeros({X.size(0), W.size(0), 3}, K_options);

    const int block_width = BLOCK_SIZE;
    const dim3 block_dim(block_width, block_width); // blockDim.x, blockDim.y
    const dim3 grid_dim((Y.size(1) + block_width - 1) / block_width, (Y.size(0) + block_width - 1) / block_width);

    AT_DISPATCH_FLOATING_TYPES(X.type(), "cuda_multi_minp_forward", ([&] {
        cuda_multi_minp_forward<scalar_t><<<grid_dim, block_dim>>>(
            X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            W.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            Y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            K.packed_accessor32<int32_t,3,torch::RestrictPtrTraits>()
        );
    }));

    return {Y, K};
}


template <typename scalar_t>
__global__ void cuda_multi_minp_backward_x(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_grad,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_grad,
    const torch::PackedTensorAccessor32<int32_t,3,torch::RestrictPtrTraits> K
) {
    const int o_r = blockIdx.y * blockDim.y + threadIdx.y;
    const int o_c = blockIdx.x * blockDim.x + threadIdx.x;

    if (o_r >= X_grad.size(0) || o_c >= X_grad.size(1)){
        return;
    }

    float out = 0.0;
    for(int dim=0; dim < 3; ++dim) {
        for(int i=0; i<Y_grad.size(1); ++i){
            if(K[o_r][i][dim] == o_c){
                out += Y_grad[o_r][i];
            }
        }
    }
    X_grad[o_r][o_c] = out;
}


void multi_minp_backward_x(
    torch::Tensor Y_grad,
    torch::Tensor X_grad, // Allocate in python
    torch::Tensor K
){
    // Can use this for the bwd pass of W as well, transpose Y_grad and K
    CHECK_INPUT(Y_grad);
    CHECK_INPUT(X_grad);
    CHECK_INPUT(K);

    assert(Y_grad.size(0) == K.size(0));
    assert(Y_grad.size(1) == K.size(1));

    const dim3 block_dim(128, 4); // blockDim.x, blockDim.y
    const dim3 grid_dim(
        (X_grad.size(1) + block_dim.x - 1) / block_dim.x,
        (X_grad.size(0) + block_dim.y - 1) / block_dim.y
    );

    AT_DISPATCH_FLOATING_TYPES(X_grad.type(), "cuda_multi_minp_backward_x", ([&] {
        cuda_multi_minp_backward_x<scalar_t><<<grid_dim, block_dim>>>(
            Y_grad.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            X_grad.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            K.packed_accessor32<int32_t,3,torch::RestrictPtrTraits>()
        );
    }));
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// MinPlus
///////////////////////////////////////////////////////////////////////////////////////////////////
#define BLOCK_SIZE 16
template <typename scalar_t>
__global__ void cuda_minp_forward(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> W,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y,
    torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> K
) {
    // Compute MaxPlus version of X @ W.T
    // X is (BATCH, D_in)
    // W is (D_out, D_in)
    // Y, K are (BATCH, D_out)
    // Y are the results and K are the max indices for bwd pass

    // Shared memory tiles (DO WE NEED AN OFFSET FOR INNER???)
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Output row and column
    const int o_r = blockIdx.y * blockDim.y + threadIdx.y;
    const int o_c = blockIdx.x * blockDim.x + threadIdx.x;

    // Block row and column
    const int r = threadIdx.y;
    const int c = threadIdx.x;

    bool in_bounds = (o_r < Y.size(0)) && (o_c < Y.size(1));

    float y_min = INFINITY;
    int k = -1;
    float sum;
    for(int i=0; i<(X.size(1) + BLOCK_SIZE - 1)/BLOCK_SIZE; ++i){
        // Load sub matrix A
        if (o_r < X.size(0) && i * BLOCK_SIZE + c < X.size(1)){
            As[r][c] = X[o_r][i * BLOCK_SIZE + c]; 
        } else {
            As[r][c] = 0;
        }

        // Load sub matrix B
        if (o_c < W.size(0) && i * BLOCK_SIZE + r < W.size(1)){
            Bs[r][c] = W[o_c][i * BLOCK_SIZE + r]; // Note W transposed, strided access problem ???
        } else {
            Bs[r][c] = 0;
        }

        __syncthreads(); // Finish load before compute
        for(int j=0; j<BLOCK_SIZE && i * BLOCK_SIZE + j < X.size(1); ++j){
            // Note that for r or c out of bounds we are still computing an (incorrect) value
            sum = As[r][j] + Bs[j][c];
            if(y_min > sum){
                y_min = sum;
                k = i * BLOCK_SIZE + j;
            }
        }
        __syncthreads(); // Finish compute before next load
    }

    if (in_bounds){
        Y[o_r][o_c] = y_min;
        K[o_r][o_c] = k;
    }
}

std::vector<torch::Tensor> minp_forward(
    torch::Tensor X,
    torch::Tensor W
){
    CHECK_INPUT(X);
    CHECK_INPUT(W);

    assert(X.size(1) == W.size(1));

    auto Y_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto Y = torch::zeros({X.size(0), W.size(0)}, Y_options);

    auto K_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto K = torch::zeros({X.size(0), W.size(0)}, K_options);

    const int block_width = BLOCK_SIZE;
    const dim3 block_dim(block_width, block_width); // blockDim.x, blockDim.y
    const dim3 grid_dim((Y.size(1) + block_width - 1) / block_width, (Y.size(0) + block_width - 1) / block_width);

    AT_DISPATCH_FLOATING_TYPES(X.type(), "cuda_minp_forward", ([&] {
        cuda_minp_forward<scalar_t><<<grid_dim, block_dim>>>(
            X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            W.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            Y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            K.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>()
        );
    }));

    return {Y, K};
}


template <typename scalar_t>
__global__ void cuda_minp_backward_x(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_grad,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_grad,
    const torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> K
) {
    const int o_r = blockIdx.y * blockDim.y + threadIdx.y;
    const int o_c = blockIdx.x * blockDim.x + threadIdx.x;

    if (o_r >= X_grad.size(0) || o_c >= X_grad.size(1)){
        return;
    }

    float out = 0.0;
    for(int i=0; i<Y_grad.size(1); ++i){
        if(K[o_r][i] == o_c){
            out += Y_grad[o_r][i];
        }
    }

    X_grad[o_r][o_c] = out;
}


void minp_backward_x(
    torch::Tensor Y_grad,
    torch::Tensor X_grad, // Allocate in python
    torch::Tensor K
){
    // Can use this for the bwd pass of W as well, transpose Y_grad and K
    CHECK_INPUT(Y_grad);
    CHECK_INPUT(X_grad);
    CHECK_INPUT(K);

    assert(Y_grad.size(0) == K.size(0));
    assert(Y_grad.size(1) == K.size(1));

    const dim3 block_dim(128, 4); // blockDim.x, blockDim.y
    const dim3 grid_dim(
        (X_grad.size(1) + block_dim.x - 1) / block_dim.x,
        (X_grad.size(0) + block_dim.y - 1) / block_dim.y
    );

    AT_DISPATCH_FLOATING_TYPES(X_grad.type(), "cuda_minp_backward_x", ([&] {
        cuda_minp_backward_x<scalar_t><<<grid_dim, block_dim>>>(
            Y_grad.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            X_grad.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            K.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>()
        );
    }));
}



///////////////////////////////////////////////////////////////////////////////////////////////////
// MaxPlus
///////////////////////////////////////////////////////////////////////////////////////////////////
#define BLOCK_SIZE 16
template <typename scalar_t>
__global__ void cuda_mp_forward(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> W,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y,
    torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> K
) {
    // Compute MaxPlus version of X @ W.T
    // X is (BATCH, D_in)
    // W is (D_out, D_in)
    // Y, K are (BATCH, D_out)
    // Y are the results and K are the max indices for bwd pass

    // Shared memory tiles (DO WE NEED AN OFFSET FOR INNER???)
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Output row and column
    const int o_r = blockIdx.y * blockDim.y + threadIdx.y;
    const int o_c = blockIdx.x * blockDim.x + threadIdx.x;

    // Block row and column
    const int r = threadIdx.y;
    const int c = threadIdx.x;

    bool in_bounds = (o_r < Y.size(0)) && (o_c < Y.size(1));

    float y_max = -INFINITY;
    int k = -1;
    float sum;
    for(int i=0; i<(X.size(1) + BLOCK_SIZE - 1)/BLOCK_SIZE; ++i){
        // Load sub matrix A
        if (o_r < X.size(0) && i * BLOCK_SIZE + c < X.size(1)){
            As[r][c] = X[o_r][i * BLOCK_SIZE + c]; 
        } else {
            As[r][c] = 0;
        }

        // Load sub matrix B
        if (o_c < W.size(0) && i * BLOCK_SIZE + r < W.size(1)){
            Bs[r][c] = W[o_c][i * BLOCK_SIZE + r]; // Note W transposed, strided access problem ???
        } else {
            Bs[r][c] = 0;
        }

        __syncthreads(); // Finish load before compute
        for(int j=0; j<BLOCK_SIZE && i * BLOCK_SIZE + j < X.size(1); ++j){
            // Note that for r or c out of bounds we are still computing an (incorrect) value
            sum = As[r][j] + Bs[j][c];
            if(y_max < sum){
                y_max = sum;
                k = i * BLOCK_SIZE + j;
            }
        }
        __syncthreads(); // Finish compute before next load
    }

    if (in_bounds){
        Y[o_r][o_c] = y_max;
        K[o_r][o_c] = k;
    }
}

std::vector<torch::Tensor> mp_forward(
    torch::Tensor X,
    torch::Tensor W
){
    CHECK_INPUT(X);
    CHECK_INPUT(W);

    assert(X.size(1) == W.size(1));

    auto Y_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto Y = torch::zeros({X.size(0), W.size(0)}, Y_options);

    auto K_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto K = torch::zeros({X.size(0), W.size(0)}, K_options);

    const int block_width = BLOCK_SIZE;
    const dim3 block_dim(block_width, block_width); // blockDim.x, blockDim.y
    const dim3 grid_dim((Y.size(1) + block_width - 1) / block_width, (Y.size(0) + block_width - 1) / block_width);

    AT_DISPATCH_FLOATING_TYPES(X.type(), "cuda_mp_forward", ([&] {
        cuda_mp_forward<scalar_t><<<grid_dim, block_dim>>>(
            X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            W.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            Y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            K.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>()
        );
    }));

    return {Y, K};
}


template <typename scalar_t>
__global__ void cuda_mp_backward_x(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_grad,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_grad,
    const torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> K
) {
    const int o_r = blockIdx.y * blockDim.y + threadIdx.y;
    const int o_c = blockIdx.x * blockDim.x + threadIdx.x;

    if (o_r >= X_grad.size(0) || o_c >= X_grad.size(1)){
        return;
    }

    float out = 0.0;
    for(int i=0; i<Y_grad.size(1); ++i){
        if(K[o_r][i] == o_c){
            out += Y_grad[o_r][i];
        }
    }

    X_grad[o_r][o_c] = out;
}


void mp_backward_x(
    torch::Tensor Y_grad,
    torch::Tensor X_grad, // Allocate in python
    torch::Tensor K
){
    // Can use this for the bwd pass of W as well, transpose Y_grad and K
    CHECK_INPUT(Y_grad);
    CHECK_INPUT(X_grad);
    CHECK_INPUT(K);

    assert(Y_grad.size(0) == K.size(0));
    assert(Y_grad.size(1) == K.size(1));

    const dim3 block_dim(128, 4); // blockDim.x, blockDim.y
    const dim3 grid_dim(
        (X_grad.size(1) + block_dim.x - 1) / block_dim.x,
        (X_grad.size(0) + block_dim.y - 1) / block_dim.y
    );

    AT_DISPATCH_FLOATING_TYPES(X_grad.type(), "cuda_mp_backward_x", ([&] {
        cuda_mp_backward_x<scalar_t><<<grid_dim, block_dim>>>(
            Y_grad.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            X_grad.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            K.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>()
        );
    }));
}


// Slow kernel with minimal memory reuse without cache hits
template <typename scalar_t>
__global__ void cuda_mp_forward_naive(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> W,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y,
    torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> K
) {
    // Compute MaxPlus version of X @ W.T
    // X is (B, D_in)
    // W is (D_out, D_in)
    const int o_r = blockIdx.y * blockDim.y + threadIdx.y;
    const int o_c = blockIdx.x * blockDim.x + threadIdx.x;

    if (o_r >= Y.size(0) || o_c >= Y.size(1)){
        return;
    }

    float y = X[o_r][0] + W[0][o_c];
    int k = 0;
    float sum;
    for(int i=1; i<X.size(1); ++i){
        sum = X[o_r][i] + W[o_c][i]; // Note W transpose
        if(y < sum){
            y = sum;
            k = i;
        }
    }

    Y[o_r][o_c] = y;
    K[o_r][o_c] = k;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// MinMaxPlusPlus
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
__global__ void cuda_mmpp_forward(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> W,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y,
    torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> K_max,
    torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> K_min
) {
    // Compute MaxPlus(X, W.T) + MinPlus(X, W.T)
    // X is (BATCH, D_in)
    // W is (D_out, D_in)
    // Y, K are (BATCH, D_out)
    // Y are the results
    // K_max/K_min are the max/min indices for bwd pass

    // Shared memory tiles (DO WE NEED AN OFFSET FOR INNER???)
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Output row and column
    const int o_r = blockIdx.y * blockDim.y + threadIdx.y;
    const int o_c = blockIdx.x * blockDim.x + threadIdx.x;

    // Block row and column
    const int r = threadIdx.y;
    const int c = threadIdx.x;

    bool in_bounds = (o_r < Y.size(0)) && (o_c < Y.size(1));

    float y_max = -INFINITY;
    float y_min = INFINITY;
    int k_max = -1;
    int k_min = -1;
    float sum;
    for(int i=0; i<(X.size(1) + BLOCK_SIZE - 1)/BLOCK_SIZE; ++i){
        // Load sub matrix A
        if (o_r < X.size(0) && i * BLOCK_SIZE + c < X.size(1)){
            As[r][c] = X[o_r][i * BLOCK_SIZE + c]; 
        } else {
            As[r][c] = 0;
        }

        // Load sub matrix B
        if (o_c < W.size(0) && i * BLOCK_SIZE + r < W.size(1)){
            Bs[r][c] = W[o_c][i * BLOCK_SIZE + r]; // Note W transposed, strided access problem ???
        } else {
            Bs[r][c] = 0;
        }

        __syncthreads(); // Finish load before compute
        for(int j=0; j<BLOCK_SIZE && i * BLOCK_SIZE + j < X.size(1); ++j){
            // Note that for r or c out of bounds we are still computing an (incorrect) value
            sum = As[r][j] + Bs[j][c];
            if(y_max < sum){
                y_max = sum;
                k_max = i * BLOCK_SIZE + j;
            }
            if(y_min > sum){
                y_min = sum;
                k_min = i * BLOCK_SIZE + j;
            }
        }
        __syncthreads(); // Finish compute before next load
    }

    if (in_bounds){
        Y[o_r][o_c] = y_max + y_min;
        K_max[o_r][o_c] = k_max;
        K_min[o_r][o_c] = k_min;
    }
}


std::vector<torch::Tensor> mmpp_forward(
    torch::Tensor X,
    torch::Tensor W
){
    CHECK_INPUT(X);
    CHECK_INPUT(W);

    assert(X.size(1) == W.size(1));

    auto Y_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto Y = torch::zeros({X.size(0), W.size(0)}, Y_options);

    auto K_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto K_max = torch::zeros({X.size(0), W.size(0)}, K_options);
    auto K_min = torch::zeros({X.size(0), W.size(0)}, K_options);

    const int block_width = BLOCK_SIZE;
    const dim3 block_dim(block_width, block_width); // blockDim.x, blockDim.y
    const dim3 grid_dim((Y.size(1) + block_width - 1) / block_width, (Y.size(0) + block_width - 1) / block_width);

    AT_DISPATCH_FLOATING_TYPES(X.type(), "cuda_mmpp_forward", ([&] {
        cuda_mmpp_forward<scalar_t><<<grid_dim, block_dim>>>(
            X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            W.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            Y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            K_max.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>(),
            K_min.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>()
        );
    }));

    return {Y, K_max, K_min};
}


template <typename scalar_t>
__global__ void cuda_mmpp_backward_x(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_grad,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_grad,
    const torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> K_max,
    const torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> K_min
) {
    const int o_r = blockIdx.y * blockDim.y + threadIdx.y;
    const int o_c = blockIdx.x * blockDim.x + threadIdx.x;

    if (o_r >= X_grad.size(0) || o_c >= X_grad.size(1)){
        return;
    }

    float out = 0.0;
    for(int i=0; i<Y_grad.size(1); ++i){
        if(K_max[o_r][i] == o_c){
            out += Y_grad[o_r][i];
        }
        if(K_min[o_r][i] == o_c){
            out += Y_grad[o_r][i];
        }
    }

    X_grad[o_r][o_c] = out;
}


void mmpp_backward_x(
    torch::Tensor Y_grad,
    torch::Tensor X_grad, // Allocate in python
    torch::Tensor K_max,
    torch::Tensor K_min
){
    // Can use this for the bwd pass of W as well, transpose Y_grad and Ks
    CHECK_INPUT(Y_grad);
    CHECK_INPUT(X_grad);
    CHECK_INPUT(K_max);
    CHECK_INPUT(K_min);

    assert(Y_grad.size(0) == K_max.size(0));
    assert(Y_grad.size(1) == K_max.size(1));
    assert(Y_grad.size(0) == K_min.size(0));
    assert(Y_grad.size(1) == K_min.size(1));

    const dim3 block_dim(128, 4); // blockDim.x, blockDim.y
    const dim3 grid_dim(
        (X_grad.size(1) + block_dim.x - 1) / block_dim.x,
        (X_grad.size(0) + block_dim.y - 1) / block_dim.y
    );

    AT_DISPATCH_FLOATING_TYPES(X_grad.type(), "cuda_mmpp_backward_x", ([&] {
        cuda_mmpp_backward_x<scalar_t><<<grid_dim, block_dim>>>(
            Y_grad.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            X_grad.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            K_max.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>(),
            K_min.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>()
        );
    }));
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Python Bindings
///////////////////////////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multi_mp_forward", &multi_mp_forward, "Multi Max forward");
    m.def("multi_mp_backward_x", &multi_mp_backward_x, "Multi Max backward X");
    m.def("multi_minp_forward", &multi_minp_forward, "Multi Min forward");
    m.def("multi_minp_backward_x", &multi_minp_backward_x, "Multi Min backward X");

    m.def("minp_forward", &minp_forward, "MinP forward");
    m.def("minp_backward_x", &minp_backward_x, "MinB backward X");
    m.def("mp_forward", &mp_forward, "MP forward");
    m.def("mp_backward_x", &mp_backward_x, "MB backward X");
    m.def("mmpp_forward", &mmpp_forward, "MMPP forward");
    m.def("mmpp_backward_x", &mmpp_backward_x, "MMPP backward X");
//   m.def("backward", &lltm_backward, "LLTM backward");
}
