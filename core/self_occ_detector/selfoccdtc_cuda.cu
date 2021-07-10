#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <math_constants.h>

namespace {
}
__global__ void self_occ_dtc_cuda_kernel(
    const torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits,size_t> epp1,
    const torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits,size_t> epp2,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> pts2dsrch_v1_batch,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> pts2d_v1_batch,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> pts2d_v2_batch,
    const torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> srh_distance,
    torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> occ_selector,
    const float minsr_dist,
    const float minoc_dist,
    const int bz,
    const int h,
    const int w
    ) {

    int x;
    int y;

    float dx;
    float dy;

    int xsign;
    int ysign;

    int xx;
    int xy;
    int yy;
    int yx;

    float D;
    int cty;

    int comx;
    int comy;

    for(int i = threadIdx.x; i < h * w; i = i + blockDim.x){
        y = i / w;
        x = i - y * w;
        if (srh_distance[blockIdx.x][y][x] > minsr_dist){
            dx = pts2dsrch_v1_batch[blockIdx.x][y][x][0] - float(x);
            dy = pts2dsrch_v1_batch[blockIdx.x][y][x][1] - float(y);

            if (dx > 0){
                xsign = 1;
            }
            else{
                xsign = -1;
            }

            if (dy > 0){
                ysign = 1;
            }
            else{
                ysign = -1;
            }

            dx = abs(dx);
            dy = abs(dy);

            if (dx > dy){
                xx = xsign;
                xy = 0;
                yx = 0;
                yy = ysign;
                }
            else{
                D = dx;
                dx = dy;
                dy = D;

                xx = 0;
                xy = ysign;
                yx = xsign;
                yy = 0;

            }
            D = 2 * dy - dx;
            cty = 0;

            for(int ctx = 0; ctx < ceil(dx) + 1; ctx = ctx + 1){
                comx = x + ctx * xx + cty * yx;
                comy = y + ctx * xy + cty * yy;
                if ((comx < 0) || (comx >= w-1) || (comy < 0) || (comy >= h-1)){
                    break;
                }
                if (D >= 0){
                    cty += 1;
                    D -= 2 * dx;
                }
                D += 2 * dy;

                if ((occ_selector[blockIdx.x][0][comy][comx] == 0) && (ctx > 0)){

                    if (
                    ((pts2d_v1_batch[blockIdx.x][y][x][0] - epp1[blockIdx.x][0]) * (float(comx) - pts2d_v1_batch[blockIdx.x][y][x][0]) + (pts2d_v1_batch[blockIdx.x][y][x][1] - epp1[blockIdx.x][1]) * (float(comy) - pts2d_v1_batch[blockIdx.x][y][x][1])) *
                    ((pts2d_v2_batch[blockIdx.x][y][x][0] - epp2[blockIdx.x][0]) * (pts2d_v2_batch[blockIdx.x][comy][comx][0] - pts2d_v2_batch[blockIdx.x][y][x][0]) + (pts2d_v2_batch[blockIdx.x][y][x][1] - epp2[blockIdx.x][1]) * (pts2d_v2_batch[blockIdx.x][comy][comx][1] - pts2d_v2_batch[blockIdx.x][y][x][1]))
                    < 0
                    ){
                        if (sqrt(pow(pts2d_v2_batch[blockIdx.x][comy][comx][0] - pts2d_v2_batch[blockIdx.x][y][x][0], 2) + pow(pts2d_v2_batch[blockIdx.x][comy][comx][1] - pts2d_v2_batch[blockIdx.x][y][x][1], 2) + 1e-10) < minoc_dist){
                            occ_selector[blockIdx.x][0][comy][comx] = 1;
                        }
                    }
                }
            }
        }
    }

    return;
    }

void self_occ_dtc_cuda(
    torch::Tensor epp1,
    torch::Tensor epp2,
    torch::Tensor pts2dsrch_v1_batch,
    torch::Tensor pts2d_v1_batch,
    torch::Tensor pts2d_v2_batch,
    torch::Tensor srh_distance,
    torch::Tensor occ_selector,
    float minsr_dist,
    float minoc_dist,
    int bz,
    int h,
    int w
    ){
      const int threads = 256;
      self_occ_dtc_cuda_kernel<<<bz, threads>>>(
            epp1.packed_accessor<float,2,torch::RestrictPtrTraits,size_t>(),
            epp2.packed_accessor<float,2,torch::RestrictPtrTraits,size_t>(),
            pts2dsrch_v1_batch.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            pts2d_v1_batch.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            pts2d_v2_batch.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            srh_distance.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
            occ_selector.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            minsr_dist,
            minoc_dist,
            bz,
            h,
            w
            );

    return;
    }