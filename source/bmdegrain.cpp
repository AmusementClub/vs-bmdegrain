#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <limits>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#ifdef __AVX2__
#include <vectorclass.h>
#include <immintrin.h>
#endif

#include <VapourSynth.h>
#include <VSHelper.h>

#include <config.h>

static VSPlugin * myself = nullptr;

template <typename T>
static inline T square(T const & x) noexcept {
    return x * x;
}

namespace {
struct Workspace {
    float * intermediate; // [radius == 0] shape: (2, height, width)
    float * denoising_patch; // shape: (group_size, block_stride) + pad (simd_lanes - 1)
    float * current_patch; // shape: (block_size, block_size) + pad (simd_lanes - 1)
    float * weights; // shape: (group_size,)
    std::vector<std::tuple<float, int, int, int>> * errors; // shape: dynamic
    std::vector<std::tuple<float, int, int>> * center_errors; // shape: dynamic
    std::vector<std::tuple<int, int>> * search_locations; // shape: dynamic
    std::vector<std::tuple<int, int>> * new_locations; // shape: dynamic
    std::vector<std::tuple<int, int>> * locations_copy; // shape: dynamic
    std::vector<std::tuple<float, int, int>> * temporal_errors; // shape: dynamic

    void init(
        int width, int height,
        int block_size, int group_size, int radius
    ) noexcept {

#ifdef __AVX2__
        constexpr int pad = 7;
#else
        constexpr int pad = 0;
#endif

        current_patch = vs_aligned_malloc<float>((square(block_size) + pad) * sizeof(float), 64);

        if (radius == 0) {
            intermediate = reinterpret_cast<float *>(std::malloc(2 * height * width * sizeof(float)));
        } else {
            intermediate = nullptr;
        }

        int m = square(block_size);
        int n = group_size;

        denoising_patch = vs_aligned_malloc<float>((m * n + pad) * sizeof(float), 64);

        weights = reinterpret_cast<float *>(std::malloc(group_size * sizeof(float)));

        errors = new std::remove_pointer_t<decltype(errors)>;
        center_errors = new std::remove_pointer_t<decltype(center_errors)>;
        search_locations = new std::remove_pointer_t<decltype(search_locations)>;
        new_locations = new std::remove_pointer_t<decltype(new_locations)>;
        locations_copy = new std::remove_pointer_t<decltype(locations_copy)>;
        temporal_errors = new std::remove_pointer_t<decltype(temporal_errors)>;
    }

    void release() noexcept {
        std::free(weights);
        weights = nullptr;

        vs_aligned_free(current_patch);
        current_patch = nullptr;

        std::free(intermediate);
        intermediate = nullptr;

        vs_aligned_free(denoising_patch);
        denoising_patch = nullptr;

        delete errors;
        errors = nullptr;

        delete center_errors;
        center_errors = nullptr;

        delete search_locations;
        search_locations = nullptr;

        delete new_locations;
        new_locations = nullptr;

        delete locations_copy;
        locations_copy = nullptr;

        delete temporal_errors;
        temporal_errors = nullptr;
    }
};

struct BMDegrainData {
    VSNodeRef * node;
    float th_sse[3];
    int block_size, block_step, group_size, bm_range;
    int radius, ps_num, ps_range;
    bool process[3];
    bool adaptive_aggregation;
    VSNodeRef * ref_node; // rclip

    std::unordered_map<std::thread::id, Workspace> workspaces;
    std::shared_mutex workspaces_lock;
};
} // namespace

#ifdef __AVX2__
static inline Vec8i make_mask(int block_size_m8) noexcept {
    static constexpr int temp[16] {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};

    return Vec8i().load(temp + 8 - block_size_m8);
}
#endif

#ifdef __AVX2__
namespace {
enum class BlockSizeInfo { Is8, Mod8, General };

struct Empty {};
}

template <BlockSizeInfo dispatch>
static inline void compute_block_distances_avx2(
    std::vector<std::tuple<float, int, int>> & errors,
    const float * VS_RESTRICT current_patch,
    const float * VS_RESTRICT neighbour_patch,
    int top, int bottom, int left, int right,
    int stride, int block_size
) noexcept {

    if constexpr (dispatch == BlockSizeInfo::Is8) {
        block_size = 8;
    }

    [[maybe_unused]] std::conditional_t<dispatch == BlockSizeInfo::General, Vec8i, Empty> mask;
    if constexpr (dispatch == BlockSizeInfo::General) {
        mask = make_mask(block_size % 8);
    }

    for (int bm_y = top; bm_y <= bottom; ++bm_y) {
        for (int bm_x = left; bm_x <= right; ++bm_x) {
            Vec8f vec_error {0.f};

            const float * VS_RESTRICT current_patchp = current_patch;
            const float * VS_RESTRICT neighbour_patchp = neighbour_patch;

            for (int patch_y = 0; patch_y < block_size; ++patch_y) {
                if constexpr (dispatch == BlockSizeInfo::Is8) {
                    Vec8f vec_current = Vec8f().load_a(current_patchp);
                    Vec8f vec_neighbour = Vec8f().load(neighbour_patchp);

                    Vec8f diff = vec_current - vec_neighbour;
                    vec_error = mul_add(diff, diff, vec_error);

                    current_patchp += 8;
                    neighbour_patchp += stride;
                } else if constexpr (dispatch == BlockSizeInfo::Mod8) {
                    for (int patch_x = 0; patch_x < block_size; patch_x += 8) {
                        Vec8f vec_current = Vec8f().load_a(current_patchp);
                        Vec8f vec_neighbour = Vec8f().load(neighbour_patchp);

                        Vec8f diff = vec_current - vec_neighbour;
                        vec_error = mul_add(diff, diff, vec_error);

                        current_patchp += 8;
                        neighbour_patchp += 8;
                    }

                    neighbour_patchp += stride - block_size;
                } else if constexpr (dispatch == BlockSizeInfo::General) {
                    for (int patch_x = 0; patch_x < (block_size & (-8)); patch_x += 8) {
                        Vec8f vec_current = Vec8f().load(current_patchp);
                        Vec8f vec_neighbour = Vec8f().load(neighbour_patchp);

                        Vec8f diff = vec_current - vec_neighbour;
                        vec_error = mul_add(diff, diff, vec_error);

                        current_patchp += 8;
                        neighbour_patchp += 8;
                    }

                    {
                        Vec8f vec_current = _mm256_maskload_ps(current_patchp, mask);
                        Vec8f vec_neighbour = _mm256_maskload_ps(neighbour_patchp, mask);

                        Vec8f diff = vec_current - vec_neighbour;
                        vec_error = mul_add(diff, diff, vec_error);

                        current_patchp += block_size % 8;
                        neighbour_patchp += stride - (block_size & (-8));
                    }
                }
            }

            float error { horizontal_add(vec_error) };

            errors.emplace_back(error, bm_x, bm_y);

            neighbour_patch++;
        }

        neighbour_patch += stride - (right - left + 1);
    }
}

template <BlockSizeInfo dispatch>
static inline void compute_block_distances_avx2(
    std::vector<std::tuple<float, int, int>> & errors,
    const float * VS_RESTRICT current_patch,
    const float * VS_RESTRICT refp,
    const std::vector<std::tuple<int, int>> & search_positions,
    int stride, int block_size
) noexcept {

    if constexpr (dispatch == BlockSizeInfo::Is8) {
        block_size = 8;
    }

    [[maybe_unused]] std::conditional_t<dispatch == BlockSizeInfo::General, Vec8i, Empty> mask;
    if constexpr (dispatch == BlockSizeInfo::General) {
        mask = make_mask(block_size % 8);
    }

    for (const auto & [bm_x, bm_y]: search_positions) {
        Vec8f vec_error {0.f};

        const float * VS_RESTRICT current_patchp = current_patch;
        const float * VS_RESTRICT neighbour_patchp = &refp[bm_y * stride + bm_x];

        for (int patch_y = 0; patch_y < block_size; ++patch_y) {
            if constexpr (dispatch == BlockSizeInfo::Is8) {
                Vec8f vec_current = Vec8f().load_a(current_patchp);
                Vec8f vec_neighbour = Vec8f().load(neighbour_patchp);

                Vec8f diff = vec_current - vec_neighbour;
                vec_error = mul_add(diff, diff, vec_error);

                current_patchp += 8;
                neighbour_patchp += stride;
            } else if constexpr (dispatch == BlockSizeInfo::Mod8) {
                for (int patch_x = 0; patch_x < block_size; patch_x += 8) {
                    Vec8f vec_current = Vec8f().load_a(current_patchp);
                    Vec8f vec_neighbour = Vec8f().load(neighbour_patchp);

                    Vec8f diff = vec_current - vec_neighbour;
                    vec_error = mul_add(diff, diff, vec_error);

                    current_patchp += 8;
                    neighbour_patchp += 8;
                }

                neighbour_patchp += stride - block_size;
            } else if constexpr (dispatch == BlockSizeInfo::General) {
                for (int patch_x = 0; patch_x < (block_size & (-8)); patch_x += 8) {
                    Vec8f vec_current = Vec8f().load(current_patchp);
                    Vec8f vec_neighbour = Vec8f().load(neighbour_patchp);

                    Vec8f diff = vec_current - vec_neighbour;
                    vec_error = mul_add(diff, diff, vec_error);

                    current_patchp += 8;
                    neighbour_patchp += 8;
                }

                {
                    Vec8f vec_current = _mm256_maskload_ps(current_patchp, mask);
                    Vec8f vec_neighbour = _mm256_maskload_ps(neighbour_patchp, mask);

                    Vec8f diff = vec_current - vec_neighbour;
                    vec_error = mul_add(diff, diff, vec_error);

                    current_patchp += block_size % 8;
                    neighbour_patchp += stride - (block_size & (-8));
                }
            }
        }

        float error { horizontal_add(vec_error) };

        errors.emplace_back(error, bm_x, bm_y);
    }
}
#endif // __AVX2__

static inline void generate_search_locations(
    const std::tuple<float, int, int> * center_positions, int num_center_positions,
    int block_size, int width, int height, int bm_range,
    std::vector<std::tuple<int, int>> & search_locations,
    std::vector<std::tuple<int, int>> & new_locations,
    std::vector<std::tuple<int, int>> & locations_copy
) noexcept {

    search_locations.clear();

    for (int i = 0; i < num_center_positions; i++) {
        const auto & [_, x, y] = center_positions[i];
        int left = std::max(x - bm_range, 0);
        int right = std::min(x + bm_range, width - block_size);
        int top = std::max(y - bm_range, 0);
        int bottom = std::min(y + bm_range, height - block_size);

        new_locations.clear();
        new_locations.reserve((bottom - top + 1) * (right - left + 1));
        for (int j = top; j <= bottom; j++) {
            for (int k = left; k <= right; k++) {
                new_locations.emplace_back(k, j);
            }
        }

        locations_copy = search_locations;

        search_locations.reserve(std::size(search_locations) + std::size(new_locations));

        search_locations.clear();

        std::set_union(
            std::cbegin(locations_copy), std::cend(locations_copy),
            std::cbegin(new_locations), std::cend(new_locations),
            std::back_inserter(search_locations),
            [](const std::tuple<int, int> & a, const std::tuple<int, int> & b) -> bool {
                auto [ax, ay] = a;
                auto [bx, by] = b;
                return ay < by || (ay == by && ax < bx);
            }
        );
    }
}

static inline void compute_block_distances(
    std::vector<std::tuple<float, int, int>> & errors,
    const float * VS_RESTRICT current_patch,
    const float * VS_RESTRICT neighbour_patch,
    int top, int bottom, int left, int right,
    int stride,
    int block_size
) noexcept {

#ifdef __AVX2__
    if (block_size == 8) {
        return compute_block_distances_avx2<BlockSizeInfo::Is8>(errors, current_patch, neighbour_patch, top, bottom, left, right, stride, block_size);
    } else if ((block_size % 8) == 0) {
        return compute_block_distances_avx2<BlockSizeInfo::Mod8>(errors, current_patch, neighbour_patch, top, bottom, left, right, stride, block_size);
    } else {
        return compute_block_distances_avx2<BlockSizeInfo::General>(errors, current_patch, neighbour_patch, top, bottom, left, right, stride, block_size);
    }
#else // __AVX2__
    for (int bm_y = top; bm_y <= bottom; ++bm_y) {
        for (int bm_x = left; bm_x <= right; ++bm_x) {
            float error = 0.f;

            const float * VS_RESTRICT current_patchp = current_patch;
            const float * VS_RESTRICT neighbour_patchp = neighbour_patch;

            for (int patch_y = 0; patch_y < block_size; ++patch_y) {
                for (int patch_x = 0; patch_x < block_size; ++patch_x) {
                    error += square(current_patchp[patch_x] - neighbour_patchp[patch_x]);
                }

                current_patchp += block_size;
                neighbour_patchp += stride;
            }

            errors.emplace_back(error, bm_x, bm_y);

            neighbour_patch++;
        }

        neighbour_patch += stride - (right - left + 1);
    }
#endif // __AVX2__
}

static inline void compute_block_distances(
    std::vector<std::tuple<float, int, int>> & errors,
    const float * VS_RESTRICT current_patch,
    const float * VS_RESTRICT refp,
    const std::vector<std::tuple<int, int>> & search_positions,
    int stride,
    int block_size
) noexcept {

#ifdef __AVX2__
    if (block_size == 8) {
        return compute_block_distances_avx2<BlockSizeInfo::Is8>(
            errors,
            current_patch, refp, search_positions, stride, block_size
        );
    } else if ((block_size % 8) == 0) {
        return compute_block_distances_avx2<BlockSizeInfo::Mod8>(
            errors,
            current_patch, refp, search_positions, stride, block_size
        );
    } else {
        return compute_block_distances_avx2<BlockSizeInfo::General>(
            errors,
            current_patch, refp, search_positions, stride, block_size
        );
    }
#else // __AVX2__
    for (const auto & [bm_x, bm_y]: search_positions) {
        float error = 0.f;

        const float * VS_RESTRICT current_patchp = current_patch;
        const float * VS_RESTRICT neighbour_patchp = &refp[bm_y * stride + bm_x];

        for (int patch_y = 0; patch_y < block_size; ++patch_y) {
            for (int patch_x = 0; patch_x < block_size; ++patch_x) {
                error += square(current_patchp[patch_x] - neighbour_patchp[patch_x]);
            }

            current_patchp += block_size;
            neighbour_patchp += stride;
        }

        errors.emplace_back(error, bm_x, bm_y);
    }
#endif // __AVX2__
}

#ifdef __AVX2__
template <BlockSizeInfo dispatch>
static inline void load_patches_avx2(
    float * VS_RESTRICT denoising_patch, int block_stride,
    const std::vector<const float *> & srcps,
    const std::vector<std::tuple<float, int, int, int>> & errors,
    int stride,
    int active_group_size,
    int block_size
) noexcept {

    if constexpr (dispatch == BlockSizeInfo::Is8) {
        block_size = 8;
    }

    [[maybe_unused]] std::conditional_t<dispatch == BlockSizeInfo::General, Vec8i, Empty> mask;
    if constexpr (dispatch == BlockSizeInfo::General) {
        mask = make_mask(block_size % 8);
    }

    assert(stride % 8 == 0);

    for (int i = 0; i < active_group_size; ++i) {
        auto [error, bm_x, bm_y, bm_t] = errors[i];

        const float * VS_RESTRICT src_patchp = &srcps[bm_t][bm_y * stride + bm_x];

        for (int patch_y = 0; patch_y < block_size; ++patch_y) {
            if constexpr (dispatch == BlockSizeInfo::Is8) {
                Vec8f vec_src = Vec8f().load(src_patchp);
                vec_src.store_a(denoising_patch);
                src_patchp += stride;
                denoising_patch += 8;
            } else if constexpr (dispatch == BlockSizeInfo::Mod8) {
                for (int patch_x = 0; patch_x < block_size; patch_x += 8) {
                    Vec8f vec_src = Vec8f().load(src_patchp);
                    vec_src.store_a(denoising_patch);
                    src_patchp += 8;
                    denoising_patch += 8;
                }

                src_patchp += stride - block_size;
            } if constexpr (dispatch == BlockSizeInfo::General) {
                for (int patch_x = 0; patch_x < (block_size & (-8)); patch_x += 8) {
                    Vec8f vec_src = Vec8f().load(src_patchp);
                    vec_src.store(denoising_patch);
                    src_patchp += 8;
                    denoising_patch += 8;
                }

                {
                    Vec8f vec_src = _mm256_maskload_ps(src_patchp, mask);
                    vec_src.store(denoising_patch); // denoising_patch is padded
                    src_patchp += stride - (block_size & (-8));
                    denoising_patch += block_size % 8;
                }
            }
        }

        if constexpr (dispatch == BlockSizeInfo::General) {
            denoising_patch += block_stride - square(block_size);
        } else {
            assert(block_stride - square(block_size) == 0);
        }
    }
}
#endif // __AVX2__

static inline void load_patches(
    float * VS_RESTRICT denoising_patch, int block_stride,
    const std::vector<const float *> & srcps,
    const std::vector<std::tuple<float, int, int, int>> & errors,
    int stride,
    int active_group_size,
    int block_size
) noexcept {

#ifdef __AVX2__
    if (block_size == 8) {
        return load_patches_avx2<BlockSizeInfo::Is8>(
            denoising_patch, block_stride,
            srcps, errors, stride,
            active_group_size, block_size
        );
    } else if ((block_size % 8) == 0) { // block_size % 8 == 0
        return load_patches_avx2<BlockSizeInfo::Mod8>(
            denoising_patch, block_stride,
            srcps, errors, stride,
            active_group_size, block_size
        );
    } else { // block_size % 8 != 0
        return load_patches_avx2<BlockSizeInfo::General>(
            denoising_patch, block_stride,
            srcps, errors, stride,
            active_group_size, block_size
        );
    }
#else // __AVX2__
    for (int i = 0, index = 0; i < active_group_size; ++i) {
        auto [error, bm_x, bm_y, bm_t] = errors[i];

        const float * VS_RESTRICT src_patchp = &srcps[bm_t][bm_y * stride + bm_x];

        for (int patch_y = 0; patch_y < block_size; ++patch_y) {
            for (int patch_x = 0; patch_x < block_size; ++patch_x) {
                float src_val = src_patchp[patch_x];

                denoising_patch[patch_x] = src_val;
            }

            src_patchp += stride;
            denoising_patch += block_size;
        }

        denoising_patch += block_stride - square(block_size);
    }
#endif // __AVX2__
}

static inline void extend_errors(
    std::vector<std::tuple<float, int, int, int>> & errors,
    const std::vector<std::tuple<float, int, int>> & spatial_errors,
    int temporal_index
) noexcept {

    errors.reserve(std::size(errors) + std::size(spatial_errors));
    for (const auto & [error, x, y] : spatial_errors) {
        errors.emplace_back(error, x, y, temporal_index);
    }
}

static inline int block_matching(
    float * VS_RESTRICT denoising_patch, int block_stride,
    std::vector<std::tuple<float, int, int, int>> & errors,
    float * VS_RESTRICT current_patch,
    const std::vector<const float *> & srcps, // length: 2 * radius + 1
    const std::vector<const float *> & refps, // length: 2 * radius + 1
    int width, int height, int stride,
    int x, int y,
    int block_size, int group_size, int bm_range,
    int ps_num, int ps_range,
    std::vector<std::tuple<float, int, int>> & center_errors,
    std::vector<std::tuple<int, int>> & search_locations,
    std::vector<std::tuple<int, int>> & new_locations,
    std::vector<std::tuple<int, int>> & locations_copy,
    std::vector<std::tuple<float, int, int>> & temporal_errors
) noexcept {

    errors.clear();
    center_errors.clear();

    auto radius = (static_cast<int>(std::size(srcps)) - 1) / 2;

    vs_bitblt(
        current_patch, block_size * sizeof(float),
        &refps[radius][y * stride + x], stride * sizeof(float),
        block_size * sizeof(float), block_size
    );

    int top = std::max(y - bm_range, 0);
    int bottom = std::min(y + bm_range, height - block_size);
    int left = std::max(x - bm_range, 0);
    int right = std::min(x + bm_range, width - block_size);

    compute_block_distances(
        center_errors,
        current_patch,
        &refps[radius][top * stride + left],
        top, bottom, left, right,
        stride, block_size
    );

    if (radius == 0) {
        extend_errors(errors, center_errors, radius);
    } else {
        int active_ps_num = std::min(
            ps_num,
            static_cast<int>(std::size(center_errors))
        );

        int active_num = std::min(
            std::max(group_size, ps_num),
            static_cast<int>(std::size(center_errors))
        );

        std::partial_sort(
            center_errors.begin(),
            center_errors.begin() + active_num,
            center_errors.end(),
            [](auto a, auto b) { return std::get<0>(a) < std::get<0>(b); }
        );
        center_errors.resize(active_num);
        extend_errors(errors, center_errors, radius);

        for (int direction = -1; direction <= 1; direction += 2) {
            temporal_errors = center_errors; // mutable

            for (int i = 1; i <= radius; i++) {
                auto temporal_index = radius + direction * i;

                generate_search_locations(
                    std::data(temporal_errors), active_ps_num,
                    block_size, width, height, ps_range,
                    search_locations, new_locations, locations_copy
                );

                temporal_errors.clear();

                compute_block_distances(
                    temporal_errors,
                    current_patch,
                    refps[temporal_index],
                    search_locations,
                    stride, block_size
                );

                auto active_temporal_num = std::min(
                    std::max(group_size, ps_num),
                    static_cast<int>(std::size(temporal_errors))
                );

                std::partial_sort(
                    temporal_errors.begin(),
                    temporal_errors.begin() + active_temporal_num,
                    temporal_errors.end(),
                    [](auto a, auto b) { return std::get<0>(a) < std::get<0>(b); }
                );
                temporal_errors.resize(active_temporal_num);
                extend_errors(errors, temporal_errors, temporal_index);
            }
        }
    }

    int active_group_size = std::min(group_size, static_cast<int>(std::size(errors)));
    std::partial_sort(
        errors.begin(),
        errors.begin() + active_group_size,
        errors.end(),
        [](auto a, auto b) { return std::get<0>(a) < std::get<0>(b); }
    );
    errors.resize(active_group_size);
    bool center = false;
    for (int i = 0; i < active_group_size; i++) {
        const auto & [_, bm_x, bm_y, bm_t] = errors[i];
        if (bm_x == x && bm_y == y && bm_t == radius) {
            center = true;
        }
    }
    if (!center) {
        errors[0] = std::make_tuple(0.0f, x, y, radius);
    }

    load_patches(
        denoising_patch, block_stride,
        srcps, errors, stride, active_group_size, block_size);

    return active_group_size;
}

static inline void patch_estimation(
    float * VS_RESTRICT output,
    float * VS_RESTRICT weights,
    const float * VS_RESTRICT denoising_patch, int block_stride,
    float & adaptive_weight,
    float th_sse,
    int block_size, int active_group_size,
    bool adaptive_aggregation
) noexcept {

    adaptive_weight = 1.0f;

    float sum_weights {};

    weights[0] = 1.0f;
    sum_weights += weights[0];

    for (int i = 1; i < active_group_size; i++) {
        float sum_squared_error {};
        for (int j = 0; j < square(block_size); j++) {
            sum_squared_error += square(denoising_patch[j] - denoising_patch[i * block_stride + j]);
        }
        if (sum_squared_error < th_sse) {
            weights[i] = (th_sse - sum_squared_error) / (th_sse + sum_squared_error);
        } else {
            weights[i] = 0.0f;
        }
        sum_weights += weights[i];
    }

    // normalize
    for (int i = 0; i < active_group_size; i++) {
        weights[i] /= sum_weights;
    }

    for (int i = 0; i < square(block_size); i++) {
        output[i] = weights[0] * denoising_patch[i];
    }

    for (int i = 1; i < active_group_size; i++) {
        for (int j = 0; j < square(block_size); j++) {
            output[j] += weights[i] * denoising_patch[i * block_stride + j];
        }
    }
}

static inline void col2im(
    float * VS_RESTRICT intermediate,
    const float * VS_RESTRICT denoising_patch, int block_stride,
    int radius, int x, int y,
    int height, int intermediate_stride,
    int block_size, int active_group_size,
    float adaptive_weight
) noexcept {

    float * VS_RESTRICT wdstp = &intermediate[(radius * 2 * height + y) * intermediate_stride + x];
    float * VS_RESTRICT weightp = &intermediate[((radius * 2 + 1) * height + y) * intermediate_stride + x];

    for (int patch_y = 0; patch_y < block_size; ++patch_y) {
        for (int patch_x = 0; patch_x < block_size; ++patch_x) {
            wdstp[patch_x] += denoising_patch[patch_x] * adaptive_weight;
            weightp[patch_x] += adaptive_weight;
        }

        wdstp += intermediate_stride;
        weightp += intermediate_stride;
        denoising_patch += block_size;
    }

    denoising_patch += block_stride - square(block_size);
}

static inline void aggregation(
    float * VS_RESTRICT dstp,
    const float * VS_RESTRICT intermediate,
    int width, int height, int stride
) noexcept {

    const float * VS_RESTRICT wdst = intermediate;
    const float * VS_RESTRICT weight = &intermediate[height * width];

    for (int y = 0; y < height; ++y) {
        int x = 0;

#ifdef __AVX2__
        const float * VS_RESTRICT vec_wdstp { wdst };
        const float * VS_RESTRICT vec_weightp { weight };
        float * VS_RESTRICT vec_dstp {dstp};

        for ( ; x < (width & (-8)); x += 8) {
            Vec8f vec_wdst = Vec8f().load(vec_wdstp);
            Vec8f vec_weight = Vec8f().load(vec_weightp);
            Vec8f vec_dst = vec_wdst * approx_recipr(vec_weight);
            vec_dst.store_a(vec_dstp);

            vec_wdstp += 8;
            vec_weightp += 8;
            vec_dstp += 8;
        }
#endif

        for ( ; x < width; ++x) {
            dstp[x] = wdst[x] / weight[x];
        }

        dstp += stride;
        wdst += width;
        weight += width;
    }
}

static void process(
    const std::vector<const VSFrameRef *> & srcs,
    const std::vector<const VSFrameRef *> & refs,
    VSFrameRef * dst,
    BMDegrainData * d,
    const VSAPI * vsapi
) noexcept {

    const auto threadId = std::this_thread::get_id();

#ifdef __AVX2__
    auto control_word = get_control_word();
    no_subnormals();
#endif

    Workspace workspace {};
    bool init = true;

    d->workspaces_lock.lock_shared();

    try {
        const auto & const_workspaces = d->workspaces;
        workspace = const_workspaces.at(threadId);
    } catch (const std::out_of_range &) {
        init = false;
    }

    d->workspaces_lock.unlock_shared();

    auto vi = vsapi->getVideoInfo(d->node);

    if (!init) {
        workspace.init(
            vi->width, vi->height,
            d->block_size, d->group_size, d->radius
        );

        d->workspaces_lock.lock();
        d->workspaces.emplace(threadId, workspace);
        d->workspaces_lock.unlock();
    }

    std::vector<std::tuple<float, int, int, int>> & errors = *workspace.errors;

    for (int plane = 0; plane < vi->format->numPlanes; plane++) {
        if (!d->process[plane]) {
            continue;
        }

        const int width = vsapi->getFrameWidth(srcs[0], plane);
        const int height = vsapi->getFrameHeight(srcs[0], plane);
        const int stride = vsapi->getStride(srcs[0], plane) / static_cast<int>(sizeof(float));
        std::vector<const float *> srcps;
        srcps.reserve(std::size(srcs));
        for (const auto & src : srcs) {
            srcps.emplace_back(reinterpret_cast<const float *>(vsapi->getReadPtr(src, plane)));
        }
        std::vector<const float *> refps;
        refps.reserve(std::size(refs));
        for (const auto & ref : refs) {
            refps.emplace_back(reinterpret_cast<const float *>(vsapi->getReadPtr(ref, plane)));
        }
        float * const VS_RESTRICT dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst, plane));

        if (d->radius == 0) {
            std::memset(workspace.intermediate, 0, 2 * height * width * sizeof(float));
        } else {
            std::memset(dstp, 0, 2 * (2 * d->radius + 1) * height * stride * sizeof(float));
        }

        int temp_r = height - d->block_size;
        int temp_c = width - d->block_size;

        for (int _y = 0; _y < temp_r + d->block_step; _y += d->block_step) {
            int y = std::min(_y, temp_r); // clamp

            for (int _x = 0; _x < temp_c + d->block_step; _x += d->block_step) {
                int x = std::min(_x, temp_c); // clamp

                int block_stride = square(d->block_size);

                int active_group_size = block_matching(
                    // outputs
                    workspace.denoising_patch, block_stride,
                    errors,
                    workspace.current_patch,
                    // inputs
                    srcps, refps, width, height, stride,
                    x, y,
                    d->block_size, d->group_size, d->bm_range,
                    d->ps_num, d->ps_range,
                    *workspace.center_errors,
                    *workspace.search_locations,
                    *workspace.new_locations,
                    *workspace.locations_copy,
                    *workspace.temporal_errors
                );

                float adaptive_weight = 1.f;
                patch_estimation(
                    workspace.current_patch, workspace.weights,
                    // outputs
                    workspace.denoising_patch, block_stride,
                    adaptive_weight,
                    // inputs
                    d->th_sse[plane],
                    d->block_size, active_group_size, d->adaptive_aggregation
                );

                if (d->radius == 0) {
                    col2im(
                        // output
                        workspace.intermediate,
                        // inputs
                        workspace.current_patch, block_stride,
                        d->radius, x, y,
                        height, width,
                        d->block_size, active_group_size, adaptive_weight
                    );
                } else {
                    col2im(
                        // output
                        dstp,
                        // inputs
                        workspace.current_patch, block_stride,
                        d->radius, x, y,
                        height, width,
                        d->block_size, active_group_size, adaptive_weight
                    );
                }
            }
        }

        if (d->radius == 0) {
            aggregation(dstp, workspace.intermediate, width, height, stride);
        }
    }

#ifdef __AVX2__
    set_control_word(control_word);
#endif
}

static void VS_CC BMDegrainRawInit(
    VSMap *in, VSMap *out, void **instanceData, VSNode *node,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    BMDegrainData * d = static_cast<BMDegrainData *>(*instanceData);

    if (d->radius > 0) {
        auto vi = *vsapi->getVideoInfo(d->node);
        vi.height *= 2 * (2 * d->radius + 1);
        vsapi->setVideoInfo(&vi, 1, node);
    } else {
        auto vi = vsapi->getVideoInfo(d->node);
        vsapi->setVideoInfo(vi, 1, node);
    }
}

static const VSFrameRef *VS_CC BMDegrainRawGetFrame(
    int n, int activationReason, void **instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto * d = static_cast<BMDegrainData *>(*instanceData);

    if (activationReason == arInitial) {
        auto vi = vsapi->getVideoInfo(d->node);

        int start_frame = std::max(n - d->radius, 0);
        int end_frame = std::min(n + d->radius, vi->numFrames - 1);

        for (int i = start_frame; i <= end_frame; ++i) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
        }
        if (d->ref_node) {
            for (int i = start_frame; i <= end_frame; ++i) {
                vsapi->requestFrameFilter(i, d->ref_node, frameCtx);
            }
        }
    } else if (activationReason == arAllFramesReady) {
        auto vi = vsapi->getVideoInfo(d->node);

        std::vector<const VSFrameRef *> srcs;
        srcs.reserve(2 * d->radius + 1);
        for (int i = -d->radius; i <= d->radius; i++) {
            auto frame_id = std::clamp(n + i, 0, vi->numFrames - 1);
            srcs.emplace_back(vsapi->getFrameFilter(frame_id, d->node, frameCtx));
        }

        std::vector<const VSFrameRef *> refs;
        if (d->ref_node) {
            refs.reserve(2 * d->radius + 1);
            for (int i = -d->radius; i <= d->radius; i++) {
                auto frame_id = std::clamp(n + i, 0, vi->numFrames - 1);
                refs.emplace_back(vsapi->getFrameFilter(frame_id, d->ref_node, frameCtx));
            }
        } else {
            refs = srcs;
        }

        const auto & center_src = srcs[d->radius];
        VSFrameRef * dst;
        if (d->radius == 0) {
            const VSFrameRef * fr[] {
                d->process[0] ? nullptr : center_src,
                d->process[1] ? nullptr : center_src,
                d->process[2] ? nullptr : center_src
            };
            const int pl[] { 0, 1, 2 };
            dst = vsapi->newVideoFrame2(vi->format, vi->width, vi->height, fr, pl, center_src, core);
        } else {
            dst = vsapi->newVideoFrame(vi->format, vi->width, 2 * (2 * d->radius + 1) * vi->height, center_src, core);
        }

        process(srcs, refs, dst, d, vsapi);

        for (const auto & src : srcs) {
            vsapi->freeFrame(src);
        }

        if (d->ref_node) {
            for (const auto & ref : refs) {
                vsapi->freeFrame(ref);
            }
        }

        return dst;
    }

    return nullptr;
}

static void VS_CC BMDegrainRawFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = static_cast<BMDegrainData *>(instanceData);

    vsapi->freeNode(d->node);

    if (d->ref_node) {
        vsapi->freeNode(d->ref_node);
    }

    for (auto & [_, workspace] : d->workspaces) {
        workspace.release();
    }

    delete d;
}

static void VS_CC BMDegrainRawCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = std::make_unique<BMDegrainData>();

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);

    auto set_error = [&](const std::string & error) -> void {
        vsapi->setError(out, ("BMDegrain: " + error).c_str());
        vsapi->freeNode(d->node);
        return ;
    };

    auto vi = vsapi->getVideoInfo(d->node);

    if (!isConstantFormat(vi) || vi->format->sampleType == stInteger ||
        (vi->format->sampleType == stFloat && vi->format->bitsPerSample != 32)
    ) {
        return set_error("only constant format 32 bit float input supported");
    }

    int error;

    for (unsigned i = 0; i < std::size(d->th_sse); i++) {
        d->th_sse[i] = static_cast<float>(vsapi->propGetFloat(in, "th_sse", i, &error));
        if (error) {
            d->th_sse[i] = (i == 0) ? 3.0f : d->th_sse[i - 1];
        }
        if (d->th_sse[i] < 0.0f) {
            return set_error("\"th_sse\" must be positive");
        }
    }

    for (unsigned i = 0; i < std::size(d->th_sse); ++i) {
        if (d->th_sse[i] < std::numeric_limits<float>::epsilon()) {
            d->process[i] = false;
        } else {
            d->process[i] = true;
            d->th_sse[i] /= 255.f;
        }
    }

    d->block_size = int64ToIntS(vsapi->propGetInt(in, "block_size", 0, &error));
    if (error) {
        // d->block_size = 6;
        d->block_size = 8; // more optimized
    } else if (d->block_size <= 0) {
        return set_error("\"th_sse\" must be positive");
    }

    for (unsigned i = 0; i < std::size(d->th_sse); ++i) {
        d->th_sse[i] *= square(d->block_size) / 64.0f;
    }

    d->block_step = int64ToIntS(vsapi->propGetInt(in, "block_step", 0, &error));
    if (error) {
        // d->block_step = 6;
        d->block_step = d->block_size; // follows the change in block_step
    } else if (d->block_step <= 0 || d->block_step > d->block_size) {
        return set_error("\"block_step\" must be positive and no larger than \"block_size\"");
    }

    d->group_size = int64ToIntS(vsapi->propGetInt(in, "group_size", 0, &error));
    if (error) {
        d->group_size = 8;
    } else if (d->group_size <= 0) {
        return set_error("\"group_size\" must be positive");
    }

    d->bm_range = int64ToIntS(vsapi->propGetInt(in, "bm_range", 0, &error));
    if (error) {
        d->bm_range = 7;
    } else if (d->bm_range < 0) {
        return set_error("\"bm_range\" must be non-negative");
    }

    d->radius = int64ToIntS(vsapi->propGetInt(in, "radius", 0, &error));
    if (error) {
        d->radius = 0;
    } else if (d->radius < 0) {
        return set_error("\"radius\" must be non-negative");
    }

    d->ps_num = int64ToIntS(vsapi->propGetInt(in, "ps_num", 0, &error));
    if (error) {
        d->ps_num = 2;
    } else if (d->ps_num <= 0) {
        return set_error("\"ps_num\" must be positive");
    }

    d->ps_range = int64ToIntS(vsapi->propGetInt(in, "ps_range", 0, &error));
    if (error) {
        d->ps_range = 4;
    } else if (d->ps_range < 0) {
        return set_error("\"ps_range\" must be non-negative");
    }

    d->adaptive_aggregation = !!vsapi->propGetInt(in, "adaptive_aggregation", 0, &error);
    if (error) {
        d->adaptive_aggregation = true;
    }

    d->ref_node = vsapi->propGetNode(in, "rclip", 0, &error);
    if (error) {
        d->ref_node = nullptr;
    } else {
        auto ref_vi = vsapi->getVideoInfo(d->ref_node);
        if (!isSameFormat(vi, ref_vi) || vi->numFrames != ref_vi->numFrames) {
            return set_error("\"rclip\" must be of the same format and number of frames as \"clip\"");
        }
    }

    VSCoreInfo core_info;
    vsapi->getCoreInfo2(core, &core_info);
    auto numThreads = core_info.numThreads;
    d->workspaces.reserve(numThreads);

    vsapi->createFilter(in, out, "BMDegrainRaw", BMDegrainRawInit, BMDegrainRawGetFrame, BMDegrainRawFree, fmParallel, 0, d.release(), core);
}

struct VAggregateData {
    VSNodeRef * node;

    VSNodeRef * src_node;
    const VSVideoInfo * src_vi;

    std::array<bool, 3> process; // th_sse != 0

    int radius;

    std::unordered_map<std::thread::id, float *> buffer;
    std::shared_mutex buffer_lock;
};

static void VS_CC VAggregateInit(
    VSMap *in, VSMap *out, void **instanceData, VSNode *node,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    auto * d = static_cast<VAggregateData *>(*instanceData);

    vsapi->setVideoInfo(d->src_vi, 1, node);
}

static const VSFrameRef *VS_CC VAggregateGetFrame(
    int n, int activationReason, void **instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto * d = static_cast<VAggregateData *>(*instanceData);

    if (activationReason == arInitial) {
        int start_frame = std::max(n - d->radius, 0);
        int end_frame = std::min(n + d->radius, d->src_vi->numFrames - 1);

        for (int i = start_frame; i <= end_frame; ++i) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
        }
        vsapi->requestFrameFilter(n, d->src_node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef * src_frame = vsapi->getFrameFilter(n, d->src_node, frameCtx);

        std::vector<const VSFrameRef *> frames;
        frames.reserve(2 * d->radius + 1);
        for (int i = n - d->radius; i <= n + d->radius; ++i) {
            auto frame_id = std::clamp(i, 0, d->src_vi->numFrames - 1);
            frames.emplace_back(vsapi->getFrameFilter(frame_id, d->node, frameCtx));
        }

        float * buffer {};
        {
            const auto thread_id = std::this_thread::get_id();
            bool init = true;

            d->buffer_lock.lock_shared();

            try {
                const auto & const_buffer = d->buffer;
                buffer = const_buffer.at(thread_id);
            } catch (const std::out_of_range &) {
                init = false;
            }

            d->buffer_lock.unlock_shared();

            if (!init) {
                assert(d->process[0] || d->src_vi->format->numPlanes > 1);

                const int max_width {
                    d->process[0] ?
                    vsapi->getFrameWidth(src_frame, 0) :
                    vsapi->getFrameWidth(src_frame, 1)
                };

                buffer = reinterpret_cast<float *>(std::malloc(2 * max_width * sizeof(float)));

                std::lock_guard _ { d->buffer_lock };
                d->buffer.emplace(thread_id, buffer);
            }
        }

        const VSFrameRef * fr[] {
            d->process[0] ? nullptr : src_frame,
            d->process[1] ? nullptr : src_frame,
            d->process[2] ? nullptr : src_frame
        };
        constexpr int pl[] { 0, 1, 2 };
        auto dst_frame = vsapi->newVideoFrame2(
            d->src_vi->format,
            d->src_vi->width, d->src_vi->height,
            fr, pl, src_frame, core);

        for (int plane = 0; plane < d->src_vi->format->numPlanes; ++plane) {
            if (d->process[plane]) {
                int plane_width = vsapi->getFrameWidth(src_frame, plane);
                int plane_height = vsapi->getFrameHeight(src_frame, plane);
                int plane_stride = vsapi->getStride(src_frame, plane) / sizeof(float);

                std::vector<const float *> srcps;
                srcps.reserve(2 * d->radius + 1);
                for (int i = 0; i < 2 * d->radius + 1; ++i) {
                    srcps.emplace_back(reinterpret_cast<const float *>(vsapi->getReadPtr(frames[i], plane)));
                }

                auto dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst_frame, plane));

                for (int y = 0; y < plane_height; ++y) {
                    std::memset(buffer, 0, 2 * plane_width * sizeof(float));
                    for (int i = 0; i < 2 * d->radius + 1; ++i) {
                        auto agg_src = srcps[i];
                        // bm3d.VAggregate implements zero padding in temporal dimension
                        // here we implements replication padding
                        agg_src += (
                            std::clamp(2 * d->radius - i, n - d->src_vi->numFrames + 1 + d->radius, n + d->radius)
                            * 2 * plane_height + y) * plane_stride;
                        for (int x = 0; x < plane_width; ++x) {
                            buffer[x] += agg_src[x];
                        }
                        agg_src += plane_height * plane_stride;
                        for (int x = 0; x < plane_width; ++x) {
                            buffer[plane_width + x] += agg_src[x];
                        }
                    }
                    for (int x = 0; x < plane_width; ++x) {
                        dstp[x] = buffer[x] / buffer[plane_width + x];
                    }
                    dstp += plane_stride;
                }
            }
        }

        for (const auto & frame : frames) {
            vsapi->freeFrame(frame);
        }
        vsapi->freeFrame(src_frame);

        return dst_frame;
    }

    return nullptr;
}

static void VS_CC VAggregateFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto * d = static_cast<VAggregateData *>(instanceData);

    for (const auto & [_, ptr] : d->buffer) {
        std::free(ptr);
    }

    vsapi->freeNode(d->src_node);
    vsapi->freeNode(d->node);

    delete d;
}

static void VS_CC VAggregateCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    {
        int error;
        bool internal = !!vsapi->propGetInt(in, "internal", 0, &error);
        if (error) {
            internal = false;
        }
        if (!internal) {
            vsapi->setError(
                out,
                "this interface is for internal use only, please use \"bmdegrain.BMDegrain()\" directly"
            );
            return ;
        }
    }

    auto d = std::make_unique<VAggregateData>();

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    auto vi = vsapi->getVideoInfo(d->node);
    d->src_node = vsapi->propGetNode(in, "src", 0, nullptr);
    d->src_vi = vsapi->getVideoInfo(d->src_node);

    d->radius = (vi->height / d->src_vi->height - 2) / 4;

    d->process.fill(false);
    int num_planes_args = vsapi->propNumElements(in, "planes");
    for (int i = 0; i < num_planes_args; ++i) {
        int plane = int64ToIntS(vsapi->propGetInt(in, "planes", i, nullptr));
        d->process[plane] = true;
    }

    VSCoreInfo core_info;
    vsapi->getCoreInfo2(core, &core_info);
    d->buffer.reserve(core_info.numThreads);

    vsapi->createFilter(
        in, out, "VAggregate",
        VAggregateInit, VAggregateGetFrame, VAggregateFree,
        fmParallel, 0, d.release(), core);
}

static void VS_CC BMDegrainCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    std::array<bool, 3> process;
    process.fill(true);

    int num_th_sse_args = vsapi->propNumElements(in, "th_sse");
    for (int i = 0; i < std::min(3, num_th_sse_args); ++i) {
        auto th_sse = vsapi->propGetFloat(in, "th_sse", i, nullptr);
        if (th_sse < std::numeric_limits<float>::epsilon()) {
            process[i] = false;
        }
    }
    if (num_th_sse_args > 0) { // num_th_sse_args may be -1
        for (int i = num_th_sse_args; i < 3; ++i) {
            process[i] = process[i - 1];
        }
    }

    bool skip = true;
    auto src = vsapi->propGetNode(in, "clip", 0, nullptr);
    auto src_vi = vsapi->getVideoInfo(src);
    for (int i = 0; i < src_vi->format->numPlanes; ++i) {
        skip &= !process[i];
    }
    if (skip) {
        vsapi->propSetNode(out, "clip", src, paReplace);
        vsapi->freeNode(src);
        return ;
    }

    auto map = vsapi->invoke(myself, "BMDegrainRaw", in);
    if (auto error = vsapi->getError(map); error) {
        vsapi->setError(out, error);
        vsapi->freeMap(map);
        vsapi->freeNode(src);
        return ;
    }

    int err;
    int radius = int64ToIntS(vsapi->propGetInt(in, "radius", 0, &err));
    if (err) {
        radius = 0;
    }
    if (radius == 0) {
        // spatial BMDegrain should handle everything itself
        auto node = vsapi->propGetNode(map, "clip", 0, nullptr);
        vsapi->freeMap(map);
        vsapi->propSetNode(out, "clip", node, paReplace);
        vsapi->freeNode(node);
        vsapi->freeNode(src);
        return ;
    }

    vsapi->propSetNode(map, "src", src, paReplace);
    vsapi->freeNode(src);

    for (int i = 0; i < 3; ++i) {
        if (process[i]) {
            vsapi->propSetInt(map, "planes", i, paAppend);
        }
    }

    vsapi->propSetInt(map, "internal", 1, paReplace);

    auto map2 = vsapi->invoke(myself, "VAggregate", map);
    vsapi->freeMap(map);
    if (auto error = vsapi->getError(map2); error) {
        vsapi->setError(out, error);
        vsapi->freeMap(map2);
        return ;
    }

    auto node = vsapi->propGetNode(map2, "clip", 0, nullptr);
    vsapi->freeMap(map2);
    vsapi->propSetNode(out, "clip", node, paReplace);
    vsapi->freeNode(node);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin
) noexcept {

    myself = plugin;

    configFunc(
        "io.github.amusementclub.bmdegrain",
        "bmdegrain", "bmdegrain",
        VAPOURSYNTH_API_VERSION, 1, plugin
    );

    constexpr auto bmdegrain_args {
        "clip:clip;"
        "th_sse:float[]:opt;"
        "block_size:int:opt;"
        "block_step:int:opt;"
        "group_size:int:opt;"
        "bm_range:int:opt;"
        "radius:int:opt;"
        "ps_num:int:opt;"
        "ps_range:int:opt;"
        //"adaptive_aggregation:int:opt;"
        "rclip:clip:opt;"
    };

    registerFunc("BMDegrainRaw", bmdegrain_args, BMDegrainRawCreate, nullptr, plugin);

    registerFunc(
        "VAggregate",
        "clip:clip;"
        "src:clip;"
        "planes:int[];"
        "internal:int:opt;",
        VAggregateCreate, nullptr, plugin);

    registerFunc("BMDegrain", bmdegrain_args, BMDegrainCreate, nullptr, plugin);

    auto getVersion = [](const VSMap *, VSMap * out, void *, VSCore *, const VSAPI *vsapi) {
        vsapi->propSetData(out, "version", VERSION, -1, paReplace);
    };
    registerFunc("Version", "", getVersion, nullptr, plugin);
}
