import numpy as np
from scipy.ndimage import label, labeled_comprehension


def remove_small_lesions_from_binary_segmentation(binary_segmentation, voxel_size, l_min=14):
    """
    Remove all lesions with less volume than `l_min` from a binary segmentation mask `binary_segmentation`.
    Args:
        binary_segmentation: `numpy.ndarray` of shape [H, W, D], with a binary lesions segmentation mask.
        voxel_size: `tuple` of length 3, with the voxel size in mm.
        l_min:  `int`, minimal volume of a lesion.
    Returns:
        Binary lesion segmentation mask (`numpy.ndarray` of shape [H, W, D])
    """

    assert type(voxel_size) == tuple, "Voxel size should be a tuple"
    assert len(voxel_size) == 3, "Voxel size should be a tuple of length 3"
    assert np.unique(binary_segmentation).tolist() == [0, 1], "Segmentation should be binary"

    labeled_seg, num_labels = label(binary_segmentation)
    label_list = np.unique(labeled_seg)
    num_elements_by_lesion = labeled_comprehension(binary_segmentation, labeled_seg, label_list, np.sum, float, 0)

    seg2 = np.zeros_like(binary_segmentation)
    for i_el, n_el in enumerate(num_elements_by_lesion):
        this_instance_indices = np.where(labeled_seg == i_el)
        this_instance_mask = np.stack(this_instance_indices, axis=1)

        size_along_x = (1 + max(this_instance_indices[0]) - min(this_instance_indices[0])) * voxel_size[0]
        size_along_y = (1 + max(this_instance_indices[1]) - min(this_instance_indices[1])) * voxel_size[1]
        size_along_z = (1 + max(this_instance_indices[2]) - min(this_instance_indices[2])) * voxel_size[2]

        # if the connected component is smaller than 3 voxels in any direction, skip it as it is not
        # clinically considered a lesion
        if size_along_x < 3 or size_along_y < 3 or size_along_z < 3:
            continue

        lesion_size = n_el * np.prod(voxel_size)
        if lesion_size > l_min:
            current_voxels = this_instance_mask
            seg2[current_voxels[:, 0],
            current_voxels[:, 1],
            current_voxels[:, 2]] = 1
    return seg2
