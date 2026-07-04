# -*- coding: utf-8 -*-
"""
PISCO Sensitivity Map Estimation in Python

This script reproduces the MATLAB code for PISCO (Parallel Imaging with Subspace-based COil sensitivity estimation)
as described in the technical report and papers cited in the original MATLAB code.

The code is converted from MATLAB to Python, ensuring line-by-line correspondence where possible,
accounting for differences in indexing (0-based in Python vs. 1-based in MATLAB), array handling,
FFT conventions, and library functions.

Key conversions:
- MATLAB's 1-based indexing is adjusted to 0-based in Python.
- FFT functions: MATLAB's fft2/ifft2 with fftshift/ifftshift are replicated using np.fft.fft2, np.fft.ifft2,
  and np.fft.fftshift/np.fft.ifftshift.
- SVD, eigenvalue decomposition: Use np.linalg.svd, np.linalg.eig.
- Random number generation: np.random.randn for randn.
- Timing: Use time.time() for tic/toc equivalents.
- Plotting: Use matplotlib.pyplot for figures, replicating MATLAB's imagesc, colormap, etc.
- Data loading: Assume 'T1_data.mat' is loaded using scipy.io.loadmat.

The script assumes the presence of 'T1_data.mat' containing 'kData' and possibly other variables.
Results should match the MATLAB output when run with the same data and parameters.

Author: Converted from MATLAB by AI Assistant
Date: Based on original MATLAB code from April 2024
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time

# Auxiliary functions
def vect(a):
    return a.flatten(order='F') 

def even_pisco(int_val):
    return int_val % 2 == 0

def C_matrix(x, N1, N2, Nc, tau, kernel_shape):
    x_reshaped = x.reshape(N1 * N2, Nc, order='F')
    
    in1, in2 = np.meshgrid(np.arange(-tau, tau + 1), np.arange(-tau, tau + 1))
    
    if kernel_shape == 1:
        mask = (in1**2 + in2**2) <= tau**2
        i = np.where(mask.flatten(order='F'))[0]
    else:
        i = np.arange(in1.size)
    
    in1 = in1.flatten(order='F')[i]
    in2 = in2.flatten(order='F')[i]
    
    patchSize = len(in1)
    
    result = np.zeros(((N1 - 2 * tau - even_pisco(N1)) * (N2 - 2 * tau - even_pisco(N2)), patchSize * Nc), dtype=x.dtype)
    
    k = 0
    for i in range(tau + even_pisco(N1), N1 - tau):
        for j in range(tau + even_pisco(N2), N2 - tau):
            ind = np.ravel_multi_index((i + in1, j + in2), (N1, N2), order='F')
            selected = x_reshaped[ind, :]
            result[k, :] = vect(selected)
            k += 1
    
    return result

def ChC_FFT_convolutions(X, N1, N2, Nc, tau, pad, kernel_shape):
    in1, in2 = np.meshgrid(np.arange(-tau, tau+1), np.arange(-tau, tau+1), indexing='xy')
    if kernel_shape == 1:
        mask = in1**2 + in2**2 <= tau**2
        i = np.where(mask.flatten(order='F'))[0]
    else:
        i = np.arange(in1.size)
    in1 = in1.flatten(order='F')[i]
    in2 = in2.flatten(order='F')[i]
    patchSize = len(in1)

    if pad:
        N1n = 2 ** int(np.ceil(np.log2(N1 + 2*tau)))
        N2n = 2 ** int(np.ceil(np.log2(N2 + 2*tau)))
    else:
        N1n = N1
        N2n = N2

    # 构建列优先线性索引矩阵 inds (patchSize, patchSize)
    row_inds = (N1n // 2) - in1[:, np.newaxis] + in1[np.newaxis, :]
    col_inds = (N2n // 2) - in2[:, np.newaxis] + in2[np.newaxis, :]
    row_inds = np.clip(row_inds, 0, N1n-1).astype(int)
    col_inds = np.clip(col_inds, 0, N2n-1).astype(int)
    inds = np.ravel_multi_index((row_inds, col_inds), (N1n, N2n), order='F')   # 列优先

    n1_freq = np.fft.fftshift(np.fft.fftfreq(N1n))
    n2_freq = np.fft.fftshift(np.fft.fftfreq(N2n))
    n2, n1 = np.meshgrid(n2_freq, n1_freq, indexing='xy')

    phaseKernel = np.exp(-1j * 2 * np.pi * (n1 * ((N1n+1)//2 + tau) + n2 * ((N2n+1)//2 + tau)))
    cphaseKernel = np.exp(-1j * 2 * np.pi * (n1 * ((N1n+1)//2) + n2 * ((N2n+1)//2)))

    x = np.fft.fft2(X, s=(N1n, N2n), axes=(0,1)) * phaseKernel[:, :, np.newaxis]

    PhP = np.zeros((patchSize, patchSize, Nc, Nc), dtype=complex)
    for q in range(Nc):
        x_rest = x[:, :, q:]                     # (N1n,N2n,Nc-q)
        x_q = x[:, :, q]                          # (N1n,N2n)
        prod = np.conj(x_rest) * x_q[:, :, np.newaxis] * cphaseKernel[:, :, np.newaxis]
        b = np.fft.ifft2(prod, axes=(0,1))        # (N1n,N2n,Nc-q)

        b = b.reshape(-1, Nc - q, order='F')      # 列优先
        b_selected = b[inds.flatten(order='F'), :]  # (patchSize*patchSize, Nc-q)
        b_selected = b_selected.reshape(patchSize, patchSize, Nc - q, order='F')

        PhP[:, :, q:, q] = b_selected
        if q < Nc - 1:
            PhP[:, :, q, q+1:] = np.conj(PhP[:, :, q+1:, q].transpose(1, 0, 2))

    PhP = PhP.transpose(0, 2, 1, 3)                # (patchSize, Nc, patchSize, Nc)
    PhP = PhP.reshape(patchSize * Nc, patchSize * Nc, order='F')
    return PhP

def nullspace_vectors_C_matrix(kCal, tau, threshold, kernel_shape, FFT_nullspace_C_calculation):
    if FFT_nullspace_C_calculation == 0:
        C = C_matrix(kCal, kCal.shape[0], kCal.shape[1], kCal.shape[2], tau, kernel_shape)
        ChC = C.conj().T @ C
    else:
        ChC = ChC_FFT_convolutions(kCal, kCal.shape[0], kCal.shape[1], kCal.shape[2], tau, 1, kernel_shape)

    _, s, vh = np.linalg.svd(ChC, full_matrices=False)
    sing = np.sqrt(np.abs(s))
    sing = sing / sing[0]
    Nvect = np.where(sing >= threshold * sing[0])[0][-1]
    U = vh.conj().T[:, Nvect+1:]      # 取右奇异向量
    return U

def G_matrices(kCal, N1, N2, tau, U, kernel_shape, FFT_interpolation, interp_zp):
    N1_cal, N2_cal, Nc = kCal.shape
    
    in1, in2 = np.meshgrid(np.arange(-tau, tau + 1), np.arange(-tau, tau + 1))
    
    flat_in1 = in1.flatten(order='F')
    flat_in2 = in2.flatten(order='F')
    if kernel_shape == 0:
        ind = np.arange(len(flat_in1))          # 取所有元素，顺序已是列优先
    else:
        mask = in1**2 + in2**2 <= tau**2
        ind = np.where(mask.flatten(order='F'))[0]   # 列优先索引
    in1 = flat_in1[ind]
    in2 = flat_in2[ind]
    
    patchSize = len(in1)
    
    in1 = in1.astype(int)
    in2 = in2.astype(int)
    
    eind = np.arange(patchSize, 0, -1) - 1
    total_size = 2 * (2 * tau + 1)
    G_flat = np.zeros(( (2*(2*tau+1)) * (2*(2*tau+1)), Nc, Nc), dtype=complex)
    W = U @ U.conj().T
    W = W.reshape(patchSize, Nc, patchSize, Nc, order='F')
    W = W.transpose(0, 1, 3, 2)   # (patchSize, Nc, Nc, patchSize)
    
    for s in range(patchSize):
        # 计算 0-based 行、列坐标
        r0 = 2 * tau + 1 + in1[eind] + in1[s]      # 形状 (patchSize,)
        c0 = 2 * tau + 1 + in2[eind] + in2[s]      # 形状 (patchSize,)
        r0 = np.clip(r0, 0, total_size-1)
        c0 = np.clip(c0, 0, total_size-1)
        linear_idx = c0 * total_size + r0 
        G_flat[linear_idx, :, :] += W[:, :, :, s]

    G = G_flat.reshape(total_size, total_size, Nc, Nc, order='F')  
    
    if FFT_interpolation == 0:
        N1_g = N1
        N2_g = N2
    else:
        if N1_cal <= N1 - interp_zp:
            N1_g = N1_cal + interp_zp
        else:
            N1_g = N1_cal
        if N2_cal <= N2 - interp_zp:
            N2_g = N2_cal + interp_zp
        else:
            N2_g = N2_cal
    
    n1 = np.fft.fftshift(np.fft.fftfreq(N1_g))
    n2 = np.fft.fftshift(np.fft.fftfreq(N2_g))
    n2, n1 = np.meshgrid(n2, n1, indexing='xy')
    phaseKernel = np.exp(-1j * 2 * np.pi * (n1 * (N1_g - 2*tau - 1) + n2 * (N2_g - 2*tau - 1)))
    G = np.fft.fft2(np.conj(G), s=(N1_g, N2_g), axes=(0,1)) * phaseKernel[:, :, np.newaxis, np.newaxis]
    G = np.fft.fftshift(G, axes=(0,1))
    
    return G



def nullspace_vectors_G_matrix(kCal, N1, N2, G, patchSize,
                               PowerIteration_G_nullspace_vectors=1,
                               M=10,
                               PowerIteration_flag_convergence=1,
                               PowerIteration_flag_auto=0,
                               FFT_interpolation=1,
                               gauss_win_param=100,
                               verbose=1):
    eps = np.finfo(float).eps

    # ---- shapes and basic init ----
    N1_g, N2_g, Nc1, Nc2 = G.shape
    assert Nc1 == Nc2, "G must be shape (N1_g, N2_g, Nc, Nc)"
    Nc = Nc1

    senseMaps = np.zeros((N1_g, N2_g, Nc), dtype=np.complex128)

    # ----- SVD branch (PowerIteration_G_nullspace_vectors == 0) -----
    if PowerIteration_G_nullspace_vectors == 0:
        eigenVal = np.zeros((N1_g, N2_g, Nc), dtype=float)
        for i in range(N1_g):
            for j in range(N2_g):
                # np.linalg.svd returns U, s, Vh where s is 1D vector, Vh is V.conj().T
                U, s, Vh = np.linalg.svd(G[i, j, :, :], full_matrices=False)
                # V_last equals V[:, -1]
                V_last = Vh[-1, :].conj()
                # phase correction: use V_last[0] (V(1,end) in MATLAB)
                senseMaps[i, j, :] = V_last * np.exp(-1j * np.angle(V_last[0]))
                # s is 1D singular values vector (length Nc)
                eigenVal[i, j, :] = np.abs(s)
        eigenVal = eigenVal / patchSize
        return senseMaps, eigenVal

    # ----- Power-iteration branch -----
    # scale G by patchSize (MATLAB: G = G/patchSize)
    G = G / patchSize

    # build identity minus G and permute dims to match MATLAB permute([1 2 4 3])
    G_null = np.zeros_like(G)
    for c in range(Nc):
        G_null[:, :, c, c] = 1.0
    G_null = G_null - G
    G_null = np.transpose(G_null, (0, 1, 3, 2))  # shape (N1_g, N2_g, Nc, Nc)

    # branch: no convergence checking, fixed M iterations
    if PowerIteration_flag_convergence == 0 and PowerIteration_flag_auto == 0:
        # init random complex maps and normalize per-pixel across channels
        senseMaps = (np.random.randn(N1_g, N2_g, Nc) + 1j * np.random.randn(N1_g, N2_g, Nc))
        norm = np.sqrt(np.sum(np.abs(senseMaps)**2, axis=2))
        norm = np.maximum(norm, eps)
        senseMaps = senseMaps / norm[:, :, np.newaxis]

        for m in range(M):
            # multiply: equivalent to MATLAB squeeze(sum(G_null .* repmat(senseMaps,[1 1 1 Nc]), 3))
            tmp = np.sum(G_null * senseMaps[:, :, np.newaxis, :], axis=3)
            # normalize
            norm = np.sqrt(np.sum(np.abs(tmp)**2, axis=2))
            norm = np.maximum(norm, eps)
            senseMaps = tmp / norm[:, :, np.newaxis]
            if m == M - 1:
                aux1 = np.sum(G_null * senseMaps[:, :, np.newaxis, :], axis=3)
                final_maps_norm = np.sqrt(np.sum(np.abs(aux1)**2, axis=2))

        eigenVal = 1.0 - final_maps_norm
        # keep eigenVal as 2D (N1_g,N2_g); matches later code paths expecting 2D when PI used
    else:
        # convergence checking path (compute first and second eigenvectors)
        senseMaps = (np.random.randn(N1_g, N2_g, Nc) + 1j * np.random.randn(N1_g, N2_g, Nc))
        eigenVec2 = (np.random.randn(N1_g, N2_g, Nc) + 1j * np.random.randn(N1_g, N2_g, Nc))

        # normalize both
        norm = np.sqrt(np.sum(np.abs(senseMaps)**2, axis=2))
        norm = np.maximum(norm, eps)
        senseMaps = senseMaps / norm[:, :, np.newaxis]

        norm2 = np.sqrt(np.sum(np.abs(eigenVec2)**2, axis=2))
        norm2 = np.maximum(norm2, eps)
        eigenVec2 = eigenVec2 / norm2[:, :, np.newaxis]

        for m in range(M):
            # power iterate both vectors
            senseMaps = np.sum(G_null * senseMaps[:, :, np.newaxis, :], axis=3)
            eigenVec2 = np.sum(G_null * eigenVec2[:, :, np.newaxis, :], axis=3)

            # normalize senseMaps
            norm = np.sqrt(np.sum(np.abs(senseMaps)**2, axis=2))
            norm = np.maximum(norm, eps)
            senseMaps = senseMaps / norm[:, :, np.newaxis]

            # orthogonalize eigenVec2 relative to senseMaps (Gram-Schmidt)
            inner_prod = np.sum(eigenVec2 * np.conj(senseMaps), axis=2)  # shape (N1_g, N2_g)
            eigenVec2 = eigenVec2 - inner_prod[:, :, np.newaxis] * senseMaps

            # normalize eigenVec2
            norm2 = np.sqrt(np.sum(np.abs(eigenVec2)**2, axis=2))
            norm2 = np.maximum(norm2, eps)
            eigenVec2 = eigenVec2 / norm2[:, :, np.newaxis]

            if m == M - 1:
                aux1 = np.sum(G_null * senseMaps[:, :, np.newaxis, :], axis=3)
                aux2 = np.sum(G_null * eigenVec2[:, :, np.newaxis, :], axis=3)
                final_maps_norm = np.sqrt(np.sum(np.abs(aux1)**2, axis=2))
                final_maps_norm2 = np.sqrt(np.sum(np.abs(aux2)**2, axis=2))

        eigen1 = final_maps_norm
        eigen2 = final_maps_norm2

        # if no FFT interpolation requested, check convergence and possibly auto-iterate
        if FFT_interpolation == 0:
            eigenVal = 1.0 - eigen1
            threshold_mask = 0.075
            support_mask = (eigenVal < threshold_mask).astype(float)

            # safe ratio: avoid divide-by-zero
            ratioEig = (eigen2 / (eigen1 + eps)) ** M
            ratio_small = support_mask * ratioEig
            th_ratio = 0.008
            ratio_small = (ratio_small > th_ratio).astype(int)
            flag_convergence_PI = np.sum(ratio_small) > 0

            if flag_convergence_PI and PowerIteration_flag_convergence == 1 and PowerIteration_flag_auto == 0:
                raise ValueError(
                    'Power Iteration might have not converged for some voxels within the support after the '
                    + str(M) + ' iterations indicated by the user. Increasing the number of iterations is recommended. '
                    'You can ignore this error by setting PowerIteration_flag_convergence = 0. '
                    'The number of needed iterations for convergence can be found automatically by setting PowerIteration_flag_auto = 1.'
                )
            if (not flag_convergence_PI) and verbose == 1:
                print('Most likely Power Iteration has converged for all the voxels within the support after the '
                      + str(M) + ' iterations indicated by the user.')

            # automatic increase of M until convergence
            if PowerIteration_flag_auto == 1 and flag_convergence_PI == 1:
                if verbose == 1:
                    print('Power Iteration auto-adjusting iterations (may take longer).')
                M_auto = M + 1
                while flag_convergence_PI:
                    senseMaps = np.sum(G_null * senseMaps[:, :, np.newaxis, :], axis=3)
                    eigenVec2 = np.sum(G_null * eigenVec2[:, :, np.newaxis, :], axis=3)

                    norm = np.sqrt(np.sum(np.abs(senseMaps)**2, axis=2))
                    norm = np.maximum(norm, eps)
                    senseMaps = senseMaps / norm[:, :, np.newaxis]

                    inner_prod = np.sum(eigenVec2 * np.conj(senseMaps), axis=2)
                    eigenVec2 = eigenVec2 - inner_prod[:, :, np.newaxis] * senseMaps

                    norm2 = np.sqrt(np.sum(np.abs(eigenVec2)**2, axis=2))
                    norm2 = np.maximum(norm2, eps)
                    eigenVec2 = eigenVec2 / norm2[:, :, np.newaxis]

                    aux1 = np.sum(G_null * senseMaps[:, :, np.newaxis, :], axis=3)
                    aux2 = np.sum(G_null * eigenVec2[:, :, np.newaxis, :], axis=3)
                    final_maps_norm = np.sqrt(np.sum(np.abs(aux1)**2, axis=2))
                    final_maps_norm2 = np.sqrt(np.sum(np.abs(aux2)**2, axis=2))
                    eigen1 = final_maps_norm
                    eigen2 = final_maps_norm2
                    eigenVal = 1.0 - eigen1

                    support_mask = (eigenVal < threshold_mask).astype(float)
                    ratioEig = (eigen2 / (eigen1 + eps)) ** M_auto
                    ratio_small = support_mask * ratioEig
                    ratio_small = (ratio_small > th_ratio).astype(int)
                    flag_convergence_PI = np.sum(ratio_small) > 0
                    M_auto += 1

                if verbose == 1:
                    print('Most likely Power Iteration has converged for all the voxels within the support. '
                          + str(M_auto) + ' iterations were needed.')

    # At this point, for PI branch we should have either eigenVal (if no convergence-check path)
    # or eigen1/eigen2 variables (if convergence-check path). For consistency downstream we
    # will keep eigenVal defined in later FFT branch.

    # ---------- FFT-based interpolation (upsample eigenvalues and map phase) ----------
    if FFT_interpolation == 1:
        N1_cal, N2_cal, _ = kCal.shape

        # build Tukey-like weight (Hann-derived window in MATLAB code formula)
        w1 = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(N1_g) / (N1_g - 1))
        w2 = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(N2_g) / (N2_g - 1))
        w_sm2d = np.outer(w1, w2)  # shape (N1_g, N2_g)

        # helper to upsample via FFT and apply window (keeps axes explicit)
        def upsample_and_window(mat2d, w2d, outN1, outN2):
            tmp = np.fft.ifftshift(mat2d, axes=(0, 1))
            tmp_k = np.fft.fft2(tmp, axes=(0, 1))
            tmp_k_shifted = np.fft.fftshift(tmp_k, axes=(0, 1))
            weighted = tmp_k_shifted * w2d
            iffted = np.fft.ifft2(weighted, s=(outN1, outN2), axes=(0, 1))
            out = np.abs(np.fft.fftshift(iffted, axes=(0, 1)))
            # normalize if nonzero
            mx = out.max()
            if mx > 0:
                out = out / mx
            return out

        # if we computed eigen1/eigen2 (in convergence path), upsample and check convergence in k-space
        if PowerIteration_G_nullspace_vectors == 1 and (PowerIteration_flag_convergence == 1 or PowerIteration_flag_auto == 1):
            # eigen1 and eigen2 must exist here (was computed above)
            auxVal = 1.0 - eigen1
            eigenVal = upsample_and_window(auxVal, w_sm2d, N1, N2)

            threshold_mask = 0.075
            support_mask = (eigenVal < threshold_mask).astype(float)

            eigen1_us = upsample_and_window(eigen1, w_sm2d, N1, N2)
            eigen2_us = upsample_and_window(eigen2, w_sm2d, N1, N2)

            ratioEig = (eigen2_us / (eigen1_us + eps)) ** M
            ratio_small = support_mask * ratioEig
            th_ratio = 0.008
            flag_convergence_PI = np.sum(ratio_small > th_ratio) > 0

            if flag_convergence_PI and PowerIteration_flag_convergence == 1 and PowerIteration_flag_auto == 0:
                raise ValueError(
                    'Power Iteration might have not converged for some voxels within the support after the '
                    + str(M) + ' iterations indicated by the user. Increasing the number of iterations is recommended.'
                )
            if (not flag_convergence_PI) and verbose == 1:
                print('Most likely Power Iteration has converged for all the voxels within the support after the '
                      + str(M) + ' iterations indicated by the user.')

            if PowerIteration_flag_auto == 1 and flag_convergence_PI == 1:
                if verbose == 1:
                    print('Power Iteration auto-adjusting iterations (FFT path).')
                M_auto = M + 1
                while flag_convergence_PI:
                    senseMaps = np.sum(G_null * senseMaps[:, :, np.newaxis, :], axis=3)
                    eigenVec2 = np.sum(G_null * eigenVec2[:, :, np.newaxis, :], axis=3)

                    norm = np.sqrt(np.sum(np.abs(senseMaps)**2, axis=2))
                    norm = np.maximum(norm, eps)
                    senseMaps = senseMaps / norm[:, :, np.newaxis]

                    inner_prod = np.sum(eigenVec2 * np.conj(senseMaps), axis=2)
                    eigenVec2 = eigenVec2 - inner_prod[:, :, np.newaxis] * senseMaps

                    norm2 = np.sqrt(np.sum(np.abs(eigenVec2)**2, axis=2))
                    norm2 = np.maximum(norm2, eps)
                    eigenVec2 = eigenVec2 / norm2[:, :, np.newaxis]

                    aux1 = np.sum(G_null * senseMaps[:, :, np.newaxis, :], axis=3)
                    aux2 = np.sum(G_null * eigenVec2[:, :, np.newaxis, :], axis=3)
                    final_maps_norm = np.sqrt(np.sum(np.abs(aux1)**2, axis=2))
                    final_maps_norm2 = np.sqrt(np.sum(np.abs(aux2)**2, axis=2))
                    eigen1 = final_maps_norm
                    eigen2 = final_maps_norm2

                    auxVal = 1.0 - eigen1
                    eigenVal = upsample_and_window(auxVal, w_sm2d, N1, N2)

                    support_mask = (eigenVal < threshold_mask).astype(float)
                    eigen1_us = upsample_and_window(eigen1, w_sm2d, N1, N2)
                    eigen2_us = upsample_and_window(eigen2, w_sm2d, N1, N2)

                    ratioEig = (eigen2_us / (eigen1_us + eps)) ** M_auto
                    ratio_small = support_mask * ratioEig
                    flag_convergence_PI = np.sum(ratio_small > th_ratio) > 0
                    M_auto += 1

                if verbose == 1:
                    print('Most likely Power Iteration has converged for all the voxels within the support. '
                          + str(M_auto) + ' iterations were needed.')

        # other cases: if PI done but no convergence checking (both flags zero),
        # or if SVD branch earlier produced eigenVal (3D), upsample accordingly.
        elif PowerIteration_G_nullspace_vectors == 1 and PowerIteration_flag_convergence == 0 and PowerIteration_flag_auto == 0:
            # eigenVal currently is 2D = 1 - final_maps_norm (from earlier branch)
            eigenVal = upsample_and_window(eigenVal, w_sm2d, N1, N2)
        elif PowerIteration_G_nullspace_vectors == 0:
            # eigenVal from SVD branch was 3D; reduce to something upsample-able.
            # Sum across channels or take smallest? MATLAB took abs(ifft2(fft2(ifftshift(eigenVal)).*w_sm))
            # Here we collapse by taking last singular value channel if present:
            # but more faithful: sum over channels before upsample
            eig2d = np.sum(eigenVal, axis=2) if 'eigenVal' in locals() and eigenVal.ndim == 3 else eigenVal
            eigenVal = upsample_and_window(eig2d, w_sm2d, N1, N2)

        # ---- apodizing window (Gaussian) ----
        try:
            from scipy.signal import windows
            w_gauss1 = windows.gausswin(N1_g, gauss_win_param)
            w_gauss2 = windows.gausswin(N2_g, gauss_win_param)
        except Exception:
            # fallback: approximate gaussian using sigma derived from gauss_win_param
            sigma1 = (N1_g - 1) / (2.0 * gauss_win_param)
            sigma2 = (N2_g - 1) / (2.0 * gauss_win_param)
            x1 = np.arange(N1_g) - (N1_g - 1) / 2.0
            x2 = np.arange(N2_g) - (N2_g - 1) / 2.0
            w_gauss1 = np.exp(-0.5 * (x1 / (sigma1 + eps))**2)
            w_gauss2 = np.exp(-0.5 * (x2 / (sigma2 + eps))**2)

        apodizing_window = np.outer(w_gauss1, w_gauss2)  # shape (N1_g, N2_g)

        # ---- build low-res image grid and place kCal centered ----
        imLowRes_cal = np.zeros((N1_g, N2_g, Nc), dtype=np.complex128)
        start1 = (N1_g - N1_cal) // 2
        start2 = (N2_g - N2_cal) // 2
        imLowRes_cal[start1:start1 + N1_cal, start2:start2 + N2_cal, :] = kCal

        # apply apodizing window and transform to image domain (MATLAB: fftshift(ifft2(ifftshift(...))))
        tmp = imLowRes_cal * apodizing_window[:, :, np.newaxis]
        tmp_ifft = np.fft.ifft2(np.fft.ifftshift(tmp, axes=(0, 1)), axes=(0, 1))
        imLowRes_cal = np.fft.fftshift(tmp_ifft, axes=(0, 1))

        # compute cim and correct phase of senseMaps
        num = np.sum(np.conj(senseMaps) * imLowRes_cal, axis=2)
        den = np.sum(np.abs(senseMaps)**2, axis=2)
        den = np.maximum(den, eps)
        cim = num / den
        senseMaps = senseMaps * np.exp(1j * np.angle(cim))[:, :, np.newaxis]

        # upsample senseMaps from (N1_g,N2_g) to (N1,N2) using same windowing in k-space
        tmp = np.fft.ifftshift(senseMaps, axes=(0, 1))
        tmp_k = np.fft.fft2(tmp, axes=(0, 1))
        tmp_k_shifted = np.fft.fftshift(tmp_k, axes=(0, 1))
        weighted = tmp_k_shifted * w_sm2d[:, :, np.newaxis]  # broadcast to channels
        iffted = np.fft.ifft2(weighted, s=(N1, N2), axes=(0, 1))
        senseMaps = np.fft.fftshift(iffted, axes=(0, 1))

    # ensure eigenVal exists for return (if not set, set a default small/zeros)
    if 'eigenVal' not in locals():
        # fallback: build a neutral eigenVal of appropriate size
        eigenVal = np.zeros((N1_g, N2_g))
    return senseMaps, eigenVal

def PISCO_senseMaps_estimation(kCal, dim_sens, tau, threshold, kernel_shape, FFT_nullspace_C_calculation, PowerIteration_G_nullspace_vectors, M, PowerIteration_flag_convergence, PowerIteration_flag_auto, FFT_interpolation, interp_zp, gauss_win_param, verbose):
    # Default values
    if tau is None or not np.isscalar(tau) or tau == 0:
        tau = 3
    if threshold is None or not np.isscalar(threshold) or threshold == 0:
        threshold = 0.05
    if kernel_shape is None or not np.isscalar(kernel_shape):
        kernel_shape = 1
    if FFT_nullspace_C_calculation is None or not np.isscalar(FFT_nullspace_C_calculation):
        FFT_nullspace_C_calculation = 1
    if PowerIteration_G_nullspace_vectors is None or not np.isscalar(PowerIteration_G_nullspace_vectors):
        PowerIteration_G_nullspace_vectors = 1
    if M is None or not np.isscalar(M) or M == 0:
        M = 10
    if PowerIteration_flag_convergence is None or not np.isscalar(PowerIteration_flag_convergence):
        PowerIteration_flag_convergence = 1
    if PowerIteration_flag_auto is None or not np.isscalar(PowerIteration_flag_auto):
        PowerIteration_flag_auto = 0
    if FFT_interpolation is None or not np.isscalar(FFT_interpolation):
        FFT_interpolation = 1
    if interp_zp is None or not np.isscalar(interp_zp) or interp_zp == 0:
        interp_zp = 24
    if gauss_win_param is None or not np.isscalar(gauss_win_param) or gauss_win_param == 0:
        gauss_win_param = 100
    if verbose is None:
        verbose = 1
    
    if verbose:
        if kernel_shape == 0:
            kernel_shape_q = 'Rectangular'
        else:
            kernel_shape_q = 'Ellipsoidal'
        if FFT_nullspace_C_calculation == 0:
            FFT_nullspace_C_calculation_q = 'No'
        else:
            FFT_nullspace_C_calculation_q = 'Yes'
        if FFT_interpolation == 0:
            FFT_interpolation_q = 'No'
        else:
            FFT_interpolation_q = 'Yes'
        if PowerIteration_G_nullspace_vectors == 0:
            PowerIteration_nullspace_vectors_q = 'No'
        else:
            PowerIteration_nullspace_vectors_q = 'Yes'
        print('Selected PISCO techniques:')
        print('========================')
        print(f'Kernel shape : {kernel_shape_q}')
        print(f'FFT-based calculation of nullspace vectors of C : {FFT_nullspace_C_calculation_q}')
        print(f'FFT-based interpolation : {FFT_interpolation_q}')
        print(f'PowerIteration-based nullspace estimation for G matrices : {PowerIteration_nullspace_vectors_q}')
        print('========================')
    
    t_null = time.time()
    
    t_null_vecs = time.time()
    U = nullspace_vectors_C_matrix(kCal, tau, threshold, kernel_shape, FFT_nullspace_C_calculation)
    t_null_vecs = time.time() - t_null_vecs
    
    if verbose:
        if FFT_nullspace_C_calculation == 0:
            aux_word = 'Calculating C first'
        else:
            aux_word = 'FFT-based direct calculation of ChC'
        print('========================')
        print('PISCO computation times (secs):')
        print('========================')
        print(f'Time nullspace vectors of C ({aux_word}) : {t_null_vecs}')
        print('========================')
    
    t_G_matrices = time.time()
    G = G_matrices(kCal, dim_sens[0], dim_sens[1], tau, U, kernel_shape, FFT_interpolation, interp_zp)
    t_G_matrices = time.time() - t_G_matrices
    
    Nc = kCal.shape[2]
    patchSize = U.shape[0] // Nc
    
    if verbose:
        print(f'Time G matrices (direct calculation): {t_G_matrices}')
        print('========================')
    
    t_null_G = time.time()
    senseMaps, eigenValues = nullspace_vectors_G_matrix(kCal, dim_sens[0], dim_sens[1], G, patchSize, PowerIteration_G_nullspace_vectors, M, PowerIteration_flag_convergence, PowerIteration_flag_auto, FFT_interpolation, gauss_win_param, verbose)
    t_null_G = time.time() - t_null_G
    
    if verbose:
        if PowerIteration_G_nullspace_vectors == 0:
            aux_word = 'Using SVD'
        else:
            aux_word = 'Using Power Iteration'
        print(f'Time nullspace vector G matrices ({aux_word}) : {t_null_G}')
        print('========================')
    
    senseMaps *= np.exp(-1j * np.angle(senseMaps[:, :, 0]))[:, :, np.newaxis]
    senseMaps /= np.sqrt(np.sum(np.abs(senseMaps)**2, axis=2))[:, :, np.newaxis]
    
    if verbose:
        print(f'Total time: {time.time() - t_null}')
        print('========================')
    
    return senseMaps, eigenValues

def mdisp(x):
   """
   Display multi-channel complex images as tiled grid.
   Input:
       x: np.ndarray, shape (N1, N2) for single coil, or (N1, N2, Nc) for Nc coils
   Output:
       2D array of shape (N1 * n_rows, N2 * n_cols) ready for imshow
   """
   x = np.asarray(x)
   
   # 2D: single coil or already flattened
   if x.ndim == 2:
       return x
   
   # 3D: multi-coil (N1, N2, Nc)
   if x.ndim != 3:
       raise ValueError(f"Expected 2D or 3D input, got shape {x.shape}")
   
   N1, N2, Nc = x.shape
   
   # Special case: 32 coils → fixed 4×8 grid (your design)
   if Nc == 32:
       n_rows, n_cols = 4, 8
   else:
       # General case: minimal rows, columns >= sqrt(Nc)
       n_cols = int(np.ceil(np.sqrt(Nc)))
       n_rows = int(np.ceil(Nc / n_cols))
   
   # Ensure grid covers all coils (n_rows * n_cols >= Nc)
   assert n_rows * n_cols >= Nc, "Grid too small"
   
   # Allocate output
   result = np.zeros((N1 * n_rows, N2 * n_cols), dtype=x.dtype)
   
   # Fill tiles (按列填充)
   for i in range(Nc):
       col = i // n_rows
       row = i % n_rows
       result[row * N1 : (row + 1) * N1,
              col * N2 : (col + 1) * N2] = x[:, :, i]
   
   return result

# Main script
if __name__ == "__main__":
    total_tic = time.time()
    plt.close('all')
    
    # Loading data
    data = scipy.io.loadmat('/home/liujunda/PISCO-python/T1_data.mat')
    kData = data['kData'].astype(complex)
    N1, N2, Nc = kData.shape
    
    plt.figure()
    # 修改1：对kData进行IFFT，需要指定axes参数，只对空间维度(0,1)做变换，保持线圈维度
    imData = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kData, axes=(0,1)), axes=(0,1)), axes=(0,1)))
    # 修改2：使用mdisp函数将3D数据(256,256,32)拼接成2D图像
    plt.imshow(mdisp(imData), cmap='gray')
    plt.axis('image')
    plt.axis('off')
    plt.title('Data in the spatial domain')
    plt.clim(0, 1e-8)
    plt.savefig('fig1_spatial.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Selection of calibration data
    cal_length = 32
    center_x = int(np.ceil(N1 / 2)) + even_pisco(N1)
    center_y = int(np.ceil(N2 / 2)) + even_pisco(N2)
    cal_index_x = np.arange(center_x - int(np.floor(cal_length / 2)), center_x + int(np.floor(cal_length / 2)) - even_pisco(cal_length))
    cal_index_y = np.arange(center_y - int(np.floor(cal_length / 2)), center_y + int(np.floor(cal_length / 2)) - even_pisco(cal_length))
    kCal = kData[cal_index_x[:, np.newaxis], cal_index_y, :]
    
    # Parameters
    dim_sens = [N1, N2]
    tau = 3
    threshold = 0.08
    M = 10
    PowerIteration_flag_convergence = None
    PowerIteration_flag_auto = 1
    interp_zp = None
    gauss_win_param = None
    kernel_shape = 1
    FFT_nullspace_C_calculation = 1
    PowerIteration_G_nullspace_vectors = 1
    FFT_interpolation = 1
    verbose = 1
    
    # Estimation
    senseMaps, eigenValues = PISCO_senseMaps_estimation(kCal, dim_sens, tau, threshold, kernel_shape, FFT_nullspace_C_calculation, PowerIteration_G_nullspace_vectors, M, PowerIteration_flag_convergence, PowerIteration_flag_auto, FFT_interpolation, interp_zp, gauss_win_param, verbose)
    
    # Support mask
    threshold_mask = 0.05
    eig_mask = np.zeros((N1, N2))
    eig_mask[eigenValues < threshold_mask] = 1
    senseMaps_masked = senseMaps * eig_mask[:, :, np.newaxis]
    
    
    
    plt.figure()
    plt.imshow(np.abs(mdisp(senseMaps)), cmap='gray')
    plt.axis('tight')
    plt.axis('image')
    plt.axis('off')
    plt.title('Estimated sensitivity maps')
    plt.savefig('fig2_senseMaps.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    plt.figure()
    plt.imshow(np.abs(mdisp(senseMaps_masked)), cmap='gray')
    plt.axis('tight')
    plt.axis('image')
    plt.axis('off')
    plt.title('Masked sensitivity maps')
    plt.savefig('fig3_masked.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    if PowerIteration_G_nullspace_vectors == 1:
        title_eig = 'Smallest eigenvalue of normalized G matrices (spatial map)'
        plt.figure()
        plt.imshow(eigenValues, cmap='gray')
        plt.colorbar()
        plt.title(title_eig)
        plt.savefig('fig4_eigenvalues.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        title_eig = 'Eigenvalues of normalized G matrices (spatial maps)'
        plt.figure()
        plt.imshow(np.abs(mdisp(eigenValues)), cmap='gray')
        plt.colorbar()
        plt.title(title_eig)
        plt.savefig('fig4_eigenvalues.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    plt.figure()
    plt.imshow(eig_mask, cmap='gray')
    plt.axis('tight')
    plt.axis('image')
    plt.title('Support mask')
    plt.savefig('fig5_mask.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Total script time: {time.time() - total_tic} seconds")
