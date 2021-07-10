#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

// CUDA forward declarations
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
    );
// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void self_occ_dtc(
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
    ) {
    CHECK_INPUT(epp1);
    CHECK_INPUT(epp2);
    CHECK_INPUT(pts2dsrch_v1_batch);
    CHECK_INPUT(pts2d_v1_batch);
    CHECK_INPUT(pts2d_v2_batch);
    CHECK_INPUT(srh_distance);
    CHECK_INPUT(occ_selector);
    self_occ_dtc_cuda(epp1, epp2, pts2dsrch_v1_batch, pts2d_v1_batch, pts2d_v2_batch, srh_distance, occ_selector, minsr_dist, minoc_dist, bz, h, w);
    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("self_occ_dtc", &self_occ_dtc, "self occlusion detection");
}
