import numpy as np

def iiqq_to_iq(iiqq: np.ndarray) -> np.ndarray:
    """
    Convert interleaved IIQQ radar data to complex IQ format.
    
    Args:
        iiqq: Input array with shape [..., N] where N is divisible by 4
        
    Returns:
        Complex array with shape [..., N/2] and dtype complex64
    """
    if iiqq.dtype == np.complex64:
        return iiqq
    
    shape = (*iiqq.shape[:-1], iiqq.shape[-1] // 2)
    iq = np.zeros(shape, dtype=np.complex64)
    iq[..., 0::2] = 1j * iiqq[..., 0::4] + iiqq[..., 2::4]
    iq[..., 1::2] = 1j * iiqq[..., 1::4] + iiqq[..., 3::4]
    return iq


def mimo(data: np.ndarray) -> np.ndarray:
    """
    Convert raw radar data to MIMO virtual array format.
    Maps (batch, doppler, tx, rx, range) -> (batch, doppler, elevation, azimuth, range).

    Args:
        data: Input array with shape [..., tx, rx, range]
              tx=3, rx=4

    Returns:
        Virtual array with shape [..., 2, 8, range]
    """
    if len(data.shape) == 5:
        # batch, doppler, tx, rx, range = data.shape
        # mimo_data = np.zeros((batch, doppler, 2, 8, range), dtype=np.complex64)
        # mimo_data[:, :, 0, 2:6, :] = data[:, :, 1, :, :]
        # mimo_data[:, :, 1, 0:4, :] = data[:, :, 0, :, :]
        # mimo_data[:, :, 1, 4:8, :] = data[:, :, 2, :, :]
        raise NotImplementedError(f"Batch operation not supported.")
    elif len(data.shape) == 4:
        doppler, tx, rx, range = data.shape
        mimo_data = np.zeros((doppler, 2, 8, range), dtype=np.complex64)
        mimo_data[:, 0, 2:6, :] = data[:, 1, :, :]
        mimo_data[:, 1, 0:4, :] = data[:, 0, :, :]
        mimo_data[:, 1, 4:8, :] = data[:, 2, :, :]
    else:
        raise ValueError(f"Expected 4D or 5D input, got shape {data.shape}")

    return mimo_data


def fft_w_shift(data: np.ndarray, dim: int = -1, shift: bool = False) -> np.ndarray:
    """
    Apply FFT to data with shift to center the zero frequency component.
    """
    fft_array = np.fft.fft(data, axis=dim)
    if shift:
        fft_array = np.fft.fftshift(fft_array, axes=dim)
    return fft_array


def range_doppler_fft(data: np.ndarray) -> np.ndarray:
    """
    Apply FFT to data in range and doppler dimensions.
    """
    range_fft = fft_w_shift(data, dim=-1, shift=False)
    doppler_fft = fft_w_shift(range_fft, dim=0, shift=True)
    return doppler_fft


def azimuth_elevation_fft(data: np.ndarray) -> np.ndarray:
    """
    Apply FFT to data in azimuth and elevation dimensions.
    """
    # Axis 1 is Elevation (size 2), Axis 2 is Azimuth (size 8)
    elevation_fft = fft_w_shift(data, dim=1, shift=True)
    azimuth_fft = fft_w_shift(elevation_fft, dim=2, shift=True)
    return azimuth_fft


def process_single_frame(iiqq: np.ndarray) -> np.ndarray:
    """
    Process a single (non-batched) frame of radar data.
    """
    iq_data = iiqq_to_iq(iiqq)
    # Maps (doppler, tx, rx, range) -> (doppler, elevation, azimuth, range).
    mimo_data = mimo(iq_data)
    # Apply range-doppler FFT
    range_doppler_fft_data = range_doppler_fft(mimo_data)
    # Apply azimuth-elevation FFT
    azimuth_elevation_fft_data = azimuth_elevation_fft(range_doppler_fft_data)
    return azimuth_elevation_fft_data
