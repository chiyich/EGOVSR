# This code is referenced from BasicSR with modifications.
# Reference: https://github.com/xinntao/BasicSR/blob/master/basicsr/data/degradations.py  # noqa
# Original licence: Copyright (c) 2020 xinntao, under the Apache 2.0 license.

import numpy as np
from scipy import special


def get_rotated_sigma_matrix(sig_x, sig_y, theta):
    """Calculate the rotated sigma matrix (two dimensional matrix).

    Args:
        sig_x (float): Standard deviation along the horizontal direction.
        sig_y (float): Standard deviation along the vertical direction.
        theta (float): Rotation in radian.

    Returns:
        ndarray: Rotated sigma matrix.
    """

    diag = np.array([[sig_x**2, 0], [0, sig_y**2]]).astype(np.float32)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]]).astype(np.float32)

    return np.matmul(rot, np.matmul(diag, rot.T))


def _mesh_grid(kernel_size):
    """Generate the mesh grid, centering at zero.

    Args:
        kernel_size (int): The size of the kernel.

    Returns:
        x_grid (ndarray): x-coordinates with shape (kernel_size, kernel_size).
        y_grid (ndarray): y-coordiantes with shape (kernel_size, kernel_size).
        xy_grid (ndarray): stacked coordinates with shape
            (kernel_size, kernel_size, 2).
    """

    range_ = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    x_grid, y_grid = np.meshgrid(range_, range_)
    xy_grid = np.hstack((x_grid.reshape((kernel_size * kernel_size, 1)),
                         y_grid.reshape(kernel_size * kernel_size,
                                        1))).reshape(kernel_size, kernel_size,
                                                     2)

    return xy_grid, x_grid, y_grid


def calculate_gaussian_pdf(sigma_matrix, grid):
    """Calculate PDF of the bivariate Gaussian distribution.

    Args:
        sigma_matrix (ndarray): The variance matrix with shape (2, 2).
        grid (ndarray): Coordinates generated by :func:`_mesh_grid`,
            with shape (K, K, 2), where K is the kernel size.

    Returns:
        kernel (ndarrray): Un-normalized kernel.
    """

    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.matmul(grid, inverse_sigma) * grid, 2))

    return kernel


def bivariate_gaussian(kernel_size,
                       sig_x,
                       sig_y=None,
                       theta=None,
                       grid=None,
                       is_isotropic=True):
    """Generate a bivariate isotropic or anisotropic Gaussian kernel.

    In isotropic mode, only `sig_x` is used. `sig_y` and `theta` are
    ignored.

    Args:
        kernel_size (int): The size of the kernel
        sig_x (float): Standard deviation along horizontal direction.
        sig_y (float | None, optional): Standard deviation along the vertical
            direction. If it is None, 'is_isotropic' must be set to True.
            Default: None.
        theta (float | None, optional): Rotation in radian. If it is None,
            'is_isotropic' must be set to True. Default: None.
        grid (ndarray, optional): Coordinates generated by :func:`_mesh_grid`,
            with shape (K, K, 2), where K is the kernel size. Default: None
        is_isotropic (bool, optional): Whether to use an isotropic kernel.
            Default: True.

    Returns:
        kernel (ndarray): normalized kernel (i.e. sum to 1).
    """

    if grid is None:
        grid, _, _ = _mesh_grid(kernel_size)

    if is_isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0,
                                                 sig_x**2]]).astype(np.float32)
    else:
        if sig_y is None:
            raise ValueError('"sig_y" cannot be None if "is_isotropic" is '
                             'False.')

        sigma_matrix = get_rotated_sigma_matrix(sig_x, sig_y, theta)

    kernel = calculate_gaussian_pdf(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)

    return kernel


def bivariate_generalized_gaussian(kernel_size,
                                   sig_x,
                                   sig_y=None,
                                   theta=None,
                                   beta=1,
                                   grid=None,
                                   is_isotropic=True):
    """Generate a bivariate generalized Gaussian kernel.

    Described in `Parameter Estimation For Multivariate Generalized
    Gaussian Distributions` by Pascal et. al (2013). In isotropic mode,
    only `sig_x` is used. `sig_y` and `theta` is ignored.

    Args:
        kernel_size (int): The size of the kernel
        sig_x (float): Standard deviation along horizontal direction
        sig_y (float | None, optional): Standard deviation along the vertical
            direction. If it is None, 'is_isotropic' must be set to True.
            Default: None.
        theta (float | None, optional): Rotation in radian. If it is None,
            'is_isotropic' must be set to True. Default: None.
        beta (float, optional): Shape parameter, beta = 1 is the normal
            distribution. Default: 1.
        grid (ndarray, optional): Coordinates generated by :func:`_mesh_grid`,
            with shape (K, K, 2), where K is the kernel size. Default: None
        is_isotropic (bool, optional): Whether to use an isotropic kernel.
            Default: True.

    Returns:
        kernel (ndarray): normalized kernel.
    """

    if grid is None:
        grid, _, _ = _mesh_grid(kernel_size)

    if is_isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0,
                                                 sig_x**2]]).astype(np.float32)
    else:
        sigma_matrix = get_rotated_sigma_matrix(sig_x, sig_y, theta)

    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(
        -0.5 *
        np.power(np.sum(np.matmul(grid, inverse_sigma) * grid, 2), beta))
    kernel = kernel / np.sum(kernel)

    return kernel


def bivariate_plateau(kernel_size,
                      sig_x,
                      sig_y,
                      theta,
                      beta,
                      grid=None,
                      is_isotropic=True):
    """Generate a plateau-like anisotropic kernel.

    This kernel has a form of 1 / (1+x^(beta)).
    Ref: https://stats.stackexchange.com/questions/203629/is-there-a-plateau-shaped-distribution  # noqa
    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.

    Args:
        kernel_size (int): The size of the kernel
        sig_x (float): Standard deviation along horizontal direction
        sig_y (float): Standard deviation along the vertical direction.
        theta (float): Rotation in radian.
        beta (float): Shape parameter, beta = 1 is the normal distribution.
        grid (ndarray, optional): Coordinates generated by :func:`_mesh_grid`,
            with shape (K, K, 2), where K is the kernel size. Default: None
        is_isotropic (bool, optional): Whether to use an isotropic kernel.
            Default: True.
    Returns:
        kernel (ndarray): normalized kernel (i.e. sum to 1).
    """
    if grid is None:
        grid, _, _ = _mesh_grid(kernel_size)

    if is_isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0,
                                                 sig_x**2]]).astype(np.float32)
    else:
        sigma_matrix = get_rotated_sigma_matrix(sig_x, sig_y, theta)

    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.reciprocal(
        np.power(np.sum(np.matmul(grid, inverse_sigma) * grid, 2), beta) + 1)
    kernel = kernel / np.sum(kernel)

    return kernel


def random_bivariate_gaussian_kernel(kernel_size,
                                     sigma_x_range,
                                     sigma_y_range,
                                     rotation_range,
                                     noise_range=None,
                                     is_isotropic=True):
    """Randomly generate bivariate isotropic or anisotropic Gaussian kernels.

    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and
    `rotation_range` is ignored.

    Args:
        kernel_size (int): The size of the kernel.
        sigma_x_range (tuple): The range of the standard deviation along the
            horizontal direction. Default: [0.6, 5]
        sigma_y_range (tuple): The range of the standard deviation along the
            vertical direction. Default: [0.6, 5]
        rotation_range (tuple): Range of rotation in radian.
        noise_range (tuple, optional): Multiplicative kernel noise.
            Default: None.
        is_isotropic (bool, optional): Whether to use an isotropic kernel.
            Default: True.

    Returns:
        kernel (ndarray): The kernel whose parameters are sampled from the
            specified range.
    """

    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] <= sigma_x_range[1], 'Wrong sigma_x_range.'

    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if is_isotropic is False:
        assert sigma_y_range[0] <= sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] <= rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    kernel = bivariate_gaussian(
        kernel_size, sigma_x, sigma_y, rotation, is_isotropic=is_isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] <= noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(
            noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)

    return kernel


def random_bivariate_generalized_gaussian_kernel(kernel_size,
                                                 sigma_x_range,
                                                 sigma_y_range,
                                                 rotation_range,
                                                 beta_range,
                                                 noise_range=None,
                                                 is_isotropic=True):
    """Randomly generate bivariate generalized Gaussian kernels.

    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and
    `rotation_range` is ignored.

    Args:
        kernel_size (int): The size of the kernel.
        sigma_x_range (tuple): The range of the standard deviation along the
            horizontal direction. Default: [0.6, 5]
        sigma_y_range (tuple): The range of the standard deviation along the
            vertical direction. Default: [0.6, 5]
        rotation_range (tuple): Range of rotation in radian.
        beta_range (float): The range of the shape parameter, beta = 1 is the
            normal distribution.
        noise_range (tuple, optional): Multiplicative kernel noise.
            Default: None.
        is_isotropic (bool, optional): Whether to use an isotropic kernel.
            Default: True.

    Returns:
        kernel (ndarray):
    """

    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] <= sigma_x_range[1], 'Wrong sigma_x_range.'

    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if is_isotropic is False:
        assert sigma_y_range[0] <= sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] <= rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    # assume beta_range[0] <= 1 <= beta_range[1]
    if np.random.uniform() <= 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel = bivariate_generalized_gaussian(
        kernel_size,
        sigma_x,
        sigma_y,
        rotation,
        beta,
        is_isotropic=is_isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] <= noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(
            noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)

    return kernel


def random_bivariate_plateau_kernel(kernel_size,
                                    sigma_x_range,
                                    sigma_y_range,
                                    rotation_range,
                                    beta_range,
                                    noise_range=None,
                                    is_isotropic=True):
    """Randomly generate bivariate plateau kernels.

    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and
    `rotation_range` is ignored.

    Args:
        kernel_size (int): The size of the kernel.
        sigma_x_range (tuple): The range of the standard deviation along the
            horizontal direction. Default: [0.6, 5]
        sigma_y_range (tuple): The range of the standard deviation along the
            vertical direction. Default: [0.6, 5]
        rotation_range (tuple): Range of rotation in radian.
        beta_range (float): The range of the shape parameter, beta = 1 is the
            normal distribution.
        noise_range (tuple, optional): Multiplicative kernel noise.
            Default: None.
        is_isotropic (bool, optional): Whether to use an isotropic kernel.
            Default: True.

    Returns:
        kernel (ndarray):
    """

    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] <= sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])

    if is_isotropic is False:
        assert sigma_y_range[0] <= sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] <= rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    # TODO: this may be not proper
    if np.random.uniform() <= 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel = bivariate_plateau(
        kernel_size,
        sigma_x,
        sigma_y,
        rotation,
        beta,
        is_isotropic=is_isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] <= noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(
            noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)

    return kernel


def random_circular_lowpass_kernel(omega_range, kernel_size, pad_to=0):
    """Generate a 2D Sinc filter.

    Reference: https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter  # noqa

    Args:
        omega_range (tuple): The cutoff frequency in radian (pi is max).
        kernel_size (int): The size of the kernel. It must be an odd number.
        pad_to (int, optional): The size of the padded kernel. It must be odd
            or zero. Default: 0.

    Returns:
        ndarray: The Sinc kernel with specified parameters.
    """
    err = np.geterr()
    np.seterr(divide='ignore', invalid='ignore')

    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    omega = np.random.uniform(omega_range[0], omega_range[-1])

    kernel = np.fromfunction(
        lambda x, y: omega * special.j1(omega * np.sqrt(
            (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)) /
        (2 * np.pi * np.sqrt((x - (kernel_size - 1) / 2)**2 +
                             (y - (kernel_size - 1) / 2)**2)),
        [kernel_size, kernel_size])
    kernel[(kernel_size - 1) // 2,
           (kernel_size - 1) // 2] = omega**2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)

    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    np.seterr(**err)

    return kernel


def random_mixed_kernels(kernel_list,
                         kernel_prob,
                         kernel_size,
                         sigma_x_range=[0.6, 5],
                         sigma_y_range=[0.6, 5],
                         rotation_range=[-np.pi, np.pi],
                         beta_gaussian_range=[0.5, 8],
                         beta_plateau_range=[1, 2],
                         omega_range=[0, np.pi],
                         noise_range=None):
    """Randomly generate a kernel.

    Args:
        kernel_list (list): A list of kernel types. Choices are
            'iso', 'aniso', 'skew', 'generalized_iso', 'generalized_aniso',
            'plateau_iso', 'plateau_aniso', 'sinc'.
        kernel_prob (list): The probability of choosing of the corresponding
            kernel.
        kernel_size (int): The size of the kernel.
        sigma_x_range (list, optional): The range of the standard deviation
            along  the horizontal direction. Default: (0.6, 5).
        sigma_y_range (list, optional): The range of the standard deviation
            along the vertical direction. Default: (0.6, 5).
        rotation_range (list, optional): Range of rotation in radian.
            Default: (-np.pi, np.pi).
        beta_gaussian_range (list, optional): The range of the shape parameter
            for generalized Gaussian. Default: (0.5, 8).
        beta_plateau_range (list, optional): The range of the shape parameter
            for plateau kernel. Default: (1, 2).
        omega_range (list, optional): The range of omega used in Sinc kernel.
            Default: (0, np.pi).
        noise_range (list, optional): Multiplicative kernel noise.
            Default: None.

    Returns:
        kernel (ndarray): The kernel whose parameters are sampled from the
            specified range.
    """

    kernel_type = np.random.choice(kernel_list, p=kernel_prob)
    if kernel_type == 'iso':
        kernel = random_bivariate_gaussian_kernel(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            noise_range=noise_range,
            is_isotropic=True)
    elif kernel_type == 'aniso':
        kernel = random_bivariate_gaussian_kernel(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            noise_range=noise_range,
            is_isotropic=False)
    elif kernel_type == 'generalized_iso':
        kernel = random_bivariate_generalized_gaussian_kernel(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            beta_gaussian_range,
            noise_range=noise_range,
            is_isotropic=True)
    elif kernel_type == 'generalized_aniso':
        kernel = random_bivariate_generalized_gaussian_kernel(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            beta_gaussian_range,
            noise_range=noise_range,
            is_isotropic=False)
    elif kernel_type == 'plateau_iso':
        kernel = random_bivariate_plateau_kernel(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            beta_plateau_range,
            noise_range=None,
            is_isotropic=True)
    elif kernel_type == 'plateau_aniso':
        kernel = random_bivariate_plateau_kernel(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            beta_plateau_range,
            noise_range=None,
            is_isotropic=False)
    elif kernel_type == 'sinc':
        kernel = random_circular_lowpass_kernel(omega_range, kernel_size)

    return kernel
