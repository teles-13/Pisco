import h5py
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# PISCO 辅助与核心算法函数定义
# ==========================================
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

    row_inds = (N1n // 2) - in1[:, np.newaxis] + in1[np.newaxis, :]
    col_inds = (N2n // 2) - in2[:, np.newaxis] + in2[np.newaxis, :]
    row_inds = np.clip(row_inds, 0, N1n-1).astype(int)
    col_inds = np.clip(col_inds, 0, N2n-1).astype(int)
    inds = np.ravel_multi_index((row_inds, col_inds), (N1n, N2n), order='F')

    n1_freq = np.fft.fftshift(np.fft.fftfreq(N1n))
    n2_freq = np.fft.fftshift(np.fft.fftfreq(N2n))
    n2, n1 = np.meshgrid(n2_freq, n1_freq, indexing='xy')

    phaseKernel = np.exp(-1j * 2 * np.pi * (n1 * ((N1n+1)//2 + tau) + n2 * ((N2n+1)//2 + tau)))
    cphaseKernel = np.exp(-1j * 2 * np.pi * (n1 * ((N1n+1)//2) + n2 * ((N2n+1)//2)))

    x = np.fft.fft2(X, s=(N1n, N2n), axes=(0,1)) * phaseKernel[:, :, np.newaxis]

    PhP = np.zeros((patchSize, patchSize, Nc, Nc), dtype=complex)
    for q in range(Nc):
        x_rest = x[:, :, q:]
        x_q = x[:, :, q]
        prod = np.conj(x_rest) * x_q[:, :, np.newaxis] * cphaseKernel[:, :, np.newaxis]
        b = np.fft.ifft2(prod, axes=(0,1))

        b = b.reshape(-1, Nc - q, order='F')
        b_selected = b[inds.flatten(order='F'), :]
        b_selected = b_selected.reshape(patchSize, patchSize, Nc - q, order='F')

        PhP[:, :, q:, q] = b_selected
        if q < Nc - 1:
            PhP[:, :, q, q+1:] = np.conj(PhP[:, :, q+1:, q].transpose(1, 0, 2))

    PhP = PhP.transpose(0, 2, 1, 3)
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
    U = vh.conj().T[:, Nvect+1:]
    return U

def G_matrices(kCal, N1, N2, tau, U, kernel_shape, FFT_interpolation, interp_zp):
    N1_cal, N2_cal, Nc = kCal.shape
    
    in1, in2 = np.meshgrid(np.arange(-tau, tau + 1), np.arange(-tau, tau + 1))
    
    flat_in1 = in1.flatten(order='F')
    flat_in2 = in2.flatten(order='F')
    if kernel_shape == 0:
        ind = np.arange(len(flat_in1))
    else:
        mask = in1**2 + in2**2 <= tau**2
        ind = np.where(mask.flatten(order='F'))[0]
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
    W = W.transpose(0, 1, 3, 2)
    
    for s in range(patchSize):
        r0 = 2 * tau + 1 + in1[eind] + in1[s]
        c0 = 2 * tau + 1 + in2[eind] + in2[s]
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

def nullspace_vectors_G_matrix(kCal, N1, N2, G, patchSize, PowerIteration_G_nullspace_vectors=1, M=10, PowerIteration_flag_convergence=1, PowerIteration_flag_auto=0, FFT_interpolation=1, gauss_win_param=100, verbose=1):
    eps = np.finfo(float).eps
    N1_g, N2_g, Nc1, Nc2 = G.shape
    Nc = Nc1
    senseMaps = np.zeros((N1_g, N2_g, Nc), dtype=np.complex128)

    if PowerIteration_G_nullspace_vectors == 0:
        eigenVal = np.zeros((N1_g, N2_g, Nc), dtype=float)
        for i in range(N1_g):
            for j in range(N2_g):
                U, s, Vh = np.linalg.svd(G[i, j, :, :], full_matrices=False)
                V_last = Vh[-1, :].conj()
                senseMaps[i, j, :] = V_last * np.exp(-1j * np.angle(V_last[0]))
                eigenVal[i, j, :] = np.abs(s)
        eigenVal = eigenVal / patchSize
        return senseMaps, eigenVal

    G = G / patchSize
    G_null = np.zeros_like(G)
    for c in range(Nc):
        G_null[:, :, c, c] = 1.0
    G_null = G_null - G
    G_null = np.transpose(G_null, (0, 1, 3, 2))

    if PowerIteration_flag_convergence == 0 and PowerIteration_flag_auto == 0:
        senseMaps = (np.random.randn(N1_g, N2_g, Nc) + 1j * np.random.randn(N1_g, N2_g, Nc))
        norm = np.sqrt(np.sum(np.abs(senseMaps)**2, axis=2))
        norm = np.maximum(norm, eps)
        senseMaps = senseMaps / norm[:, :, np.newaxis]
        for m in range(M):
            tmp = np.sum(G_null * senseMaps[:, :, np.newaxis, :], axis=3)
            norm = np.sqrt(np.sum(np.abs(tmp)**2, axis=2))
            norm = np.maximum(norm, eps)
            senseMaps = tmp / norm[:, :, np.newaxis]
            if m == M - 1:
                aux1 = np.sum(G_null * senseMaps[:, :, np.newaxis, :], axis=3)
                final_maps_norm = np.sqrt(np.sum(np.abs(aux1)**2, axis=2))
        eigenVal = 1.0 - final_maps_norm
    else:
        senseMaps = (np.random.randn(N1_g, N2_g, Nc) + 1j * np.random.randn(N1_g, N2_g, Nc))
        eigenVec2 = (np.random.randn(N1_g, N2_g, Nc) + 1j * np.random.randn(N1_g, N2_g, Nc))
        norm = np.sqrt(np.sum(np.abs(senseMaps)**2, axis=2))
        norm = np.maximum(norm, eps)
        senseMaps = senseMaps / norm[:, :, np.newaxis]
        norm2 = np.sqrt(np.sum(np.abs(eigenVec2)**2, axis=2))
        norm2 = np.maximum(norm2, eps)
        eigenVec2 = eigenVec2 / norm2[:, :, np.newaxis]

        for m in range(M):
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
            if m == M - 1:
                aux1 = np.sum(G_null * senseMaps[:, :, np.newaxis, :], axis=3)
                aux2 = np.sum(G_null * eigenVec2[:, :, np.newaxis, :], axis=3)
                final_maps_norm = np.sqrt(np.sum(np.abs(aux1)**2, axis=2))
                final_maps_norm2 = np.sqrt(np.sum(np.abs(aux2)**2, axis=2))
        eigen1 = final_maps_norm
        eigen2 = final_maps_norm2

        if FFT_interpolation == 0:
            eigenVal = 1.0 - eigen1
            threshold_mask = 0.075
            support_mask = (eigenVal < threshold_mask).astype(float)
            ratioEig = (eigen2 / (eigen1 + eps)) ** M
            ratio_small = support_mask * ratioEig
            th_ratio = 0.008
            ratio_small = (ratio_small > th_ratio).astype(int)
            flag_convergence_PI = np.sum(ratio_small) > 0

            if PowerIteration_flag_auto == 1 and flag_convergence_PI == 1:
                if verbose == 1: print('Power Iteration auto-adjusting iterations (may take longer).')
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

    if FFT_interpolation == 1:
        N1_cal, N2_cal, _ = kCal.shape
        w1 = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(N1_g) / (N1_g - 1))
        w2 = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(N2_g) / (N2_g - 1))
        w_sm2d = np.outer(w1, w2)

        def upsample_and_window(mat2d, w2d, outN1, outN2):
            tmp = np.fft.ifftshift(mat2d, axes=(0, 1))
            tmp_k = np.fft.fft2(tmp, axes=(0, 1))
            tmp_k_shifted = np.fft.fftshift(tmp_k, axes=(0, 1))
            weighted = tmp_k_shifted * w2d
            iffted = np.fft.ifft2(weighted, s=(outN1, outN2), axes=(0, 1))
            out = np.abs(np.fft.fftshift(iffted, axes=(0, 1)))
            mx = out.max()
            if mx > 0: out = out / mx
            return out

        if PowerIteration_G_nullspace_vectors == 1 and (PowerIteration_flag_convergence == 1 or PowerIteration_flag_auto == 1):
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

            if PowerIteration_flag_auto == 1 and flag_convergence_PI == 1:
                if verbose == 1: print('Power Iteration auto-adjusting iterations (FFT path).')
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
        elif PowerIteration_G_nullspace_vectors == 1 and PowerIteration_flag_convergence == 0 and PowerIteration_flag_auto == 0:
            eigenVal = upsample_and_window(eigenVal, w_sm2d, N1, N2)
        elif PowerIteration_G_nullspace_vectors == 0:
            eig2d = np.sum(eigenVal, axis=2) if 'eigenVal' in locals() and eigenVal.ndim == 3 else eigenVal
            eigenVal = upsample_and_window(eig2d, w_sm2d, N1, N2)

        try:
            from scipy.signal import windows
            w_gauss1 = windows.gausswin(N1_g, gauss_win_param)
            w_gauss2 = windows.gausswin(N2_g, gauss_win_param)
        except Exception:
            sigma1 = (N1_g - 1) / (2.0 * gauss_win_param)
            sigma2 = (N2_g - 1) / (2.0 * gauss_win_param)
            x1 = np.arange(N1_g) - (N1_g - 1) / 2.0
            x2 = np.arange(N2_g) - (N2_g - 1) / 2.0
            w_gauss1 = np.exp(-0.5 * (x1 / (sigma1 + eps))**2)
            w_gauss2 = np.exp(-0.5 * (x2 / (sigma2 + eps))**2)

        apodizing_window = np.outer(w_gauss1, w_gauss2)
        imLowRes_cal = np.zeros((N1_g, N2_g, Nc), dtype=np.complex128)
        start1 = (N1_g - N1_cal) // 2
        start2 = (N2_g - N2_cal) // 2
        imLowRes_cal[start1:start1 + N1_cal, start2:start2 + N2_cal, :] = kCal

        tmp = imLowRes_cal * apodizing_window[:, :, np.newaxis]
        tmp_ifft = np.fft.ifft2(np.fft.ifftshift(tmp, axes=(0, 1)), axes=(0, 1))
        imLowRes_cal = np.fft.fftshift(tmp_ifft, axes=(0, 1))

        num = np.sum(np.conj(senseMaps) * imLowRes_cal, axis=2)
        den = np.sum(np.abs(senseMaps)**2, axis=2)
        den = np.maximum(den, eps)
        cim = num / den
        senseMaps = senseMaps * np.exp(1j * np.angle(cim))[:, :, np.newaxis]

        tmp = np.fft.ifftshift(senseMaps, axes=(0, 1))
        tmp_k = np.fft.fft2(tmp, axes=(0, 1))
        tmp_k_shifted = np.fft.fftshift(tmp_k, axes=(0, 1))
        weighted = tmp_k_shifted * w_sm2d[:, :, np.newaxis]
        iffted = np.fft.ifft2(weighted, s=(N1, N2), axes=(0, 1))
        senseMaps = np.fft.fftshift(iffted, axes=(0, 1))

    if 'eigenVal' not in locals():
        eigenVal = np.zeros((N1_g, N2_g))
    return senseMaps, eigenVal

def PISCO_senseMaps_estimation(kCal, dim_sens, tau, threshold, kernel_shape, FFT_nullspace_C_calculation, PowerIteration_G_nullspace_vectors, M, PowerIteration_flag_convergence, PowerIteration_flag_auto, FFT_interpolation, interp_zp, gauss_win_param, verbose):
    if tau is None or not np.isscalar(tau) or tau == 0: tau = 3
    if threshold is None or not np.isscalar(threshold) or threshold == 0: threshold = 0.05
    if kernel_shape is None or not np.isscalar(kernel_shape): kernel_shape = 1
    if FFT_nullspace_C_calculation is None or not np.isscalar(FFT_nullspace_C_calculation): FFT_nullspace_C_calculation = 1
    if PowerIteration_G_nullspace_vectors is None or not np.isscalar(PowerIteration_G_nullspace_vectors): PowerIteration_G_nullspace_vectors = 1
    if M is None or not np.isscalar(M) or M == 0: M = 10
    if PowerIteration_flag_convergence is None or not np.isscalar(PowerIteration_flag_convergence): PowerIteration_flag_convergence = 1
    if PowerIteration_flag_auto is None or not np.isscalar(PowerIteration_flag_auto): PowerIteration_flag_auto = 0
    if FFT_interpolation is None or not np.isscalar(FFT_interpolation): FFT_interpolation = 1
    if interp_zp is None or not np.isscalar(interp_zp) or interp_zp == 0: interp_zp = 24
    if gauss_win_param is None or not np.isscalar(gauss_win_param) or gauss_win_param == 0: gauss_win_param = 100
    if verbose is None: verbose = 1
    
    if verbose: print('Selected PISCO techniques...')
    t_null = time.time()
    
    t_null_vecs = time.time()
    U = nullspace_vectors_C_matrix(kCal, tau, threshold, kernel_shape, FFT_nullspace_C_calculation)
    t_null_vecs = time.time() - t_null_vecs
    if verbose: print(f'Time nullspace vectors of C : {t_null_vecs}')
    
    t_G_matrices = time.time()
    G = G_matrices(kCal, dim_sens[0], dim_sens[1], tau, U, kernel_shape, FFT_interpolation, interp_zp)
    t_G_matrices = time.time() - t_G_matrices
    if verbose: print(f'Time G matrices: {t_G_matrices}')
    
    Nc = kCal.shape[2]
    patchSize = U.shape[0] // Nc
    
    t_null_G = time.time()
    senseMaps, eigenValues = nullspace_vectors_G_matrix(kCal, dim_sens[0], dim_sens[1], G, patchSize, PowerIteration_G_nullspace_vectors, M, PowerIteration_flag_convergence, PowerIteration_flag_auto, FFT_interpolation, gauss_win_param, verbose)
    t_null_G = time.time() - t_null_G
    if verbose: print(f'Time nullspace vector G matrices : {t_null_G}')
    
    senseMaps *= np.exp(-1j * np.angle(senseMaps[:, :, 0]))[:, :, np.newaxis]
    senseMaps /= np.sqrt(np.sum(np.abs(senseMaps)**2, axis=2))[:, :, np.newaxis]
    if verbose: print(f'Total estimation time: {time.time() - t_null}')
    
    return senseMaps, eigenValues

def mdisp(x):
   x = np.asarray(x)
   if x.ndim == 2: return x
   N1, N2, Nc = x.shape
   if Nc == 32:
       n_rows, n_cols = 4, 8
   else:
       n_cols = int(np.ceil(np.sqrt(Nc)))
       n_rows = int(np.ceil(Nc / n_cols))
   result = np.zeros((N1 * n_rows, N2 * n_cols), dtype=x.dtype)
   for i in range(Nc):
       col = i // n_rows
       row = i % n_rows
       result[row * N1 : (row + 1) * N1, col * N2 : (col + 1) * N2] = x[:, :, i]
   return result

# ==========================================
# 主程序：加载 fastMRI 数据并运行估计
# ==========================================
if __name__ == "__main__":
    total_tic = time.time()
    
    # --- 1. 数据读取与预处理 ---
    file_path = '/home/liujunda/data_fastmri_brain_train/multicoil_train/file_brain_AXFLAIR_200_6002425.h5'
    print(f"正在读取文件: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # 直接提取第 8 个切片，即索引为 7 的数据
        kspace_slice = f['kspace'][7] 
        
    # fastMRI 的维度是 (Nc, N1, N2)，PISCO 需要 (N1, N2, Nc)
    kData = np.transpose(kspace_slice, (1, 2, 0))
    N1, N2, Nc = kData.shape
    print(f"提取切片完成，数据形状 (N1, N2, Nc): {kData.shape}")
    
    # 可视化原始数据的图像域
    imData = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kData, axes=(0,1)), axes=(0,1)), axes=(0,1)))
    plt.figure(figsize=(10, 6))
    plt.imshow(mdisp(imData), cmap='gray')
    plt.axis('off')
    plt.title('Data in spatial domain (Slice Index 7)')
    plt.show()
    
    # --- 2. 选取校准区域 (ACS) ---
    cal_length = 32
    center_x = int(np.ceil(N1 / 2)) + even_pisco(N1)
    center_y = int(np.ceil(N2 / 2)) + even_pisco(N2)
    cal_index_x = np.arange(center_x - int(np.floor(cal_length / 2)), center_x + int(np.floor(cal_length / 2)) - even_pisco(cal_length))
    cal_index_y = np.arange(center_y - int(np.floor(cal_length / 2)), center_y + int(np.floor(cal_length / 2)) - even_pisco(cal_length))
    kCal = kData[cal_index_x[:, np.newaxis], cal_index_y, :]
    
    # --- 3. 配置参数与灵敏度图估计 ---
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
    
    print("\n开始 PISCO 算法估计...")
    senseMaps, eigenValues = PISCO_senseMaps_estimation(kCal, dim_sens, tau, threshold, kernel_shape, FFT_nullspace_C_calculation, PowerIteration_G_nullspace_vectors, M, PowerIteration_flag_convergence, PowerIteration_flag_auto, FFT_interpolation, interp_zp, gauss_win_param, verbose)
    
    # --- 4. 生成 Mask 与结果展示 ---
    threshold_mask = 0.05
    eig_mask = np.zeros((N1, N2))
    eig_mask[eigenValues < threshold_mask] = 1
    senseMaps_masked = senseMaps * eig_mask[:, :, np.newaxis]
    
    plt.figure(figsize=(12, 8))
    plt.imshow(np.abs(mdisp(senseMaps)), cmap='gray')
    plt.axis('off')
    plt.title('Estimated Sensitivity Maps')
    plt.show()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(np.abs(mdisp(senseMaps_masked)), cmap='gray')
    plt.axis('off')
    plt.title('Masked Sensitivity Maps')
    plt.show()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    im1 = ax1.imshow(eigenValues, cmap='gray')
    ax1.set_title('Smallest eigenvalue of norm G')
    fig.colorbar(im1, ax=ax1)
    
    ax2.imshow(eig_mask, cmap='gray')
    ax2.set_title('Support Mask')
    plt.show()
    
    print(f"\n全部运行完毕，总耗时: {time.time() - total_tic:.2f} 秒")
