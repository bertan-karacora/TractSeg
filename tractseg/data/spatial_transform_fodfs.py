import bonndit as bd
import numpy as np

from batchgenerators.transforms.abstract_transforms import AbstractTransform
import batchgenerators.augmentations as bgaug


def get_inner(a, b):
    """Vectorized version of bonndit's tensor dot function."""
    assert len(a) == len(b)
    order = bd.utils.tensor.get_order(a[0])
    multiplicities = np.asarray(bd.utils.tensor.MULTIPLIER[order])

    products = np.sum(a * b * multiplicities[:, np.newaxis, np.newaxis], axis=0)

    return products


def get_norm(a):
    """Vectorized version of bonndit's tensor norm function."""
    # Using Frobenius norm
    norms = np.sqrt(get_inner(a, a))
    return norms


def normalize_fodfs(fodfs):
    fodfs_normalized = fodfs / get_norm(fodfs)[np.newaxis, :, :]
    return fodfs_normalized


def create_matrix_tensor_rotation(matR, order):
    """
    Creates rotation matrix to be applied as rotation (i.e., just a change of basis)
    to vectorized symmetric tensors of the given order.
    """
    # Cartesian product. Keep order and reshape by transposing in the end for correct ordering.
    ind_all = np.indices((3,) * order).reshape(order, -1).T
    ind_sym = np.asarray(bd.utils.tensor.INDEX[order])
    # Keep these as lists for easy index retrieval
    ind_all_cumulative = [bd.utils.tensor.index_count(i) for i in ind_all]
    ind_sym_cumulative = bd.utils.tensor.CINDEX[order]
    ind_all2sym = np.asarray([ind_sym_cumulative.index(i) for i in ind_all_cumulative])

    l = bd.utils.tensor.LENGTH[order]
    matT = np.zeros((l, l))
    for i, ind_component in enumerate(ind_sym):
        coeffs_matR = matR[np.repeat(ind_component[None, :], ind_all.shape[0], axis=0), ind_all]
        coeffs_tensor = np.prod(coeffs_matR, axis=1)
        matT[i, :] = np.bincount(ind_all2sym, weights=coeffs_tensor)

    return matT


def create_rotation_matrix(angle_x, angle_y, angle_z):
    matR = np.identity(3)
    matR = bgaug.utils.create_matrix_rotation_x_3d(angle_x, matR)
    matR = bgaug.utils.create_matrix_rotation_y_3d(angle_y, matR)
    matR = bgaug.utils.create_matrix_rotation_z_3d(angle_z, matR)

    return matR


def rotate_fodfs(fodfs, angle_x, angle_y, angle_z):
    matR = create_rotation_matrix(angle_x, angle_y, angle_z)
    order = bd.utils.tensor.get_order(fodfs[:, 0, 0])
    matT = create_matrix_tensor_rotation(matR, order)

    fodfs_rotated = np.tensordot(matT, fodfs, axes=([1], [0]))

    return fodfs_rotated


def deform_coords(coords, range_alpha, range_sigma):
    alpha = np.random.uniform(range_alpha[0], range_alpha[1])
    sigma = np.random.uniform(range_sigma[0], range_sigma[1])
    coords = bgaug.utils.elastic_deform_coordinates(coords, alpha, sigma)
    return coords


def rotate_coords(coords, range_rotation):
    angle = np.random.uniform(range_rotation[0], range_rotation[1])
    coords = bgaug.utils.rotate_coords_2d(coords, angle)
    return coords, angle


def scale_coords(coords, range_scale):
    # First decide between downscaling and upscaling, then sample scale.
    if range_scale[0] < 1 and np.random.random() < 0.5:
        scale = np.random.uniform(range_scale[0], 1)
    else:
        scale = np.random.uniform(np.max([range_scale[0], 1]), range_scale[1])
    coords = bgaug.utils.scale_coords(coords, scale)

    return coords


def add_center_coords(coords, random_crop, center_dist, shape_features):
    for d in range(2):
        if random_crop:
            ctr = np.random.uniform(center_dist[d], shape_features[d] - center_dist[d])
        else:
            ctr = int(np.round(shape_features[d] / 2.0))
        coords[d] += ctr
    return coords


def augment_spatial(
    features,
    labels,
    direction_slicing,
    shape_input,
    random_crop,
    dist_center,
    do_elastic_deform,
    range_alpha,
    range_sigma,
    prob_elastic,
    do_rotation,
    range_rotation,
    prob_rotation,
    do_scale,
    range_scale,
    prob_scale,
    order_features,
    border_mode_features,
    border_cval_features,
    order_labels,
    border_mode_labels,
    border_cval_labels,
):
    features_augmented = np.zeros((features.shape[0], features.shape[1], shape_input[0], shape_input[1]), dtype=np.float32)
    labels_augmented = np.zeros((labels.shape[0], labels.shape[1], shape_input[0], shape_input[1]), dtype=np.float32)

    for sample_id in range(features.shape[0]):
        coords = bgaug.utils.create_zero_centered_coordinate_mesh(shape_input)
        modified_coords = False
        angle = None

        if do_elastic_deform and np.random.uniform() < prob_elastic:
            coords = deform_coords(coords, range_alpha, range_sigma)
            modified_coords = True

        if do_rotation and np.random.uniform() < prob_rotation:
            coords, angle = rotate_coords(coords, range_rotation)
            modified_coords = True

        if do_scale and np.random.uniform() < prob_scale:
            coords = scale_coords(coords, range_scale)
            modified_coords = True

        if modified_coords:
            coords = add_center_coords(coords, random_crop, dist_center, features.shape[2:])
            for channel_id in range(features.shape[1]):
                features_augmented[sample_id, channel_id] = bgaug.utils.interpolate_img(
                    features[sample_id, channel_id],
                    coords,
                    order_features,
                    border_mode_features,
                    cval=border_cval_features,
                )

            for channel_id in range(labels.shape[1]):
                labels_augmented[sample_id, channel_id] = bgaug.utils.interpolate_img(
                    labels[sample_id, channel_id],
                    coords,
                    order_labels,
                    border_mode_labels,
                    cval=border_cval_labels,
                    is_seg=True,
                )
        else:
            if random_crop:
                margin = [dist_center[d] - shape_input[d] // 2 for d in range(2)]
                f, l = bgaug.crop_and_pad_augmentations.random_crop_aug(features[sample_id], labels[sample_id], shape_input, margin)
            else:
                f, l = bgaug.crop_and_pad_augmentations.center_crop_aug(features[sample_id], shape_input, labels[sample_id])
            features_augmented[sample_id] = f[0]
            labels_augmented[sample_id] = l[0]

        if angle is not None:
            angle_x, angle_y, angle_z = 0, 0, 0

            if direction_slicing == 0:
                angle_x = angle
            elif direction_slicing == 1:
                angle_y = angle
            elif direction_slicing == 2:
                angle_z = angle
            else:
                raise ValueError("Invalid direction_slicing passed as argument.")

            features_augmented[sample_id] = rotate_fodfs(features_augmented[sample_id], angle_x, angle_y, angle_z)

    return features_augmented, labels_augmented


class SpatialTransformFodfs(AbstractTransform):
    """Rotation, deformation, scaling, cropping. Computational time scales only with shape_input, not with input patch size or type of augmentations used.
    Internally, this transform will use a coordinate grid of shape shape_input to which the transformations are
    applied. Interpolation on the image data will only be done at the very end.

    Args:
        shape_input (tuple/list/ndarray of int): Output patch size

        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use shape_input//2.
        This only applies when random_crop=True

        do_elastic_deform (bool): Whether or not to apply elastic deformation

        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval

        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval

        do_rotation (bool): Whether or not to apply rotation

        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!

        do_scale (bool): Whether or not to apply scaling

        scale (tuple of float): scale range ; scale is randomly sampled from interval

        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates

        border_cval_data: If border_mode_data=constant, what value to use?

        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates

        border_mode_seg: How to treat border pixels in seg? see scipy.ndimage.map_coordinates

        border_cval_seg: If border_mode_seg=constant, what value to use?

        order_seg: Order of interpolation for seg. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (for example if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in [0, 1, 2])

        random_crop: True: do a random crop of size shape_input and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of size shape_input
    """

    def __init__(
        self,
        shape_input,
        patch_center_dist_from_border=[30, 30],
        do_elastic_deform=True,
        alpha=(0.0, 1000.0),
        sigma=(10.0, 13.0),
        do_rotation=True,
        angle_x=(0, 2 * np.pi),
        # Unused but needed to keep it consistent with batchgenerators.
        angle_y=(0, 2 * np.pi),
        angle_z=(0, 2 * np.pi),
        do_scale=True,
        scale=(0.75, 1.25),
        border_mode_data="nearest",
        border_cval_data=0,
        order_data=3,
        border_mode_seg="constant",
        border_cval_seg=0,
        order_seg=0,
        random_crop=True,
        p_el_per_sample=1,
        p_scale_per_sample=1,
        p_rot_per_sample=1,
    ):
        # This is bad but necessary to keep the same call signature as batchgenerators' spatial transform API.
        self.shape_input = shape_input
        self.random_crop = random_crop
        self.dist_center = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.range_alpha = alpha
        self.range_sigma = sigma
        self.prob_elastic = p_el_per_sample
        self.do_rotation = do_rotation
        self.range_rotation = angle_x
        self.prob_rotation = p_rot_per_sample
        self.do_scale = do_scale
        self.range_scale = scale
        self.prob_scale = p_scale_per_sample
        self.order_features = order_data
        self.border_mode_features = border_mode_data
        self.border_cval_features = border_cval_data
        self.order_labels = order_seg
        self.border_mode_labels = border_mode_seg
        self.border_cval_labels = border_cval_seg

    def __call__(self, **dict_data):
        features = dict_data.get("data")
        labels = dict_data.get("seg")
        direction_slicing = dict_data.get("direction_slicing")

        dict_data["data"], dict_data["seg"] = augment_spatial(
            features,
            labels,
            direction_slicing,
            self.shape_input,
            self.random_crop,
            self.dist_center,
            self.do_elastic_deform,
            self.range_alpha,
            self.range_sigma,
            self.prob_elastic,
            self.do_rotation,
            self.range_rotation,
            self.prob_rotation,
            self.do_scale,
            self.range_scale,
            self.prob_scale,
            self.order_features,
            self.border_mode_features,
            self.border_cval_features,
            self.order_labels,
            self.border_mode_labels,
            self.border_cval_labels,
        )

        return dict_data
