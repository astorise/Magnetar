/* Minimal real reference CPU kernel: row-major single-precision matmul
 * out[m,n] = a[m,k] @ b[k,n]. Deliberately straightforward (no tiling/
 * vectorization) -- this is a reference/correctness baseline, not a
 * performance kernel, compiled for real with gcc to produce a genuine
 * ELF shared object used as the first Kernel Exchange Bundle's embedded
 * compiled-kernel blob.
 */
void magnetar_reference_matmul_f32(
    const float *a, const float *b, float *out,
    int m, int n, int k
) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}
