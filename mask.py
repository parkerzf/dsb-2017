#
#   Copyright 2017 Anil Thomas
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
"""
Segmenting lung CT scans
"""
import os
import numpy as np
import settings
from skimage import measure
from scipy import ndimage


def get_mask(image, uid):
    mask = np.array(image > -320, dtype=np.int8)
    # Set the edges to zeros. This is to connect the air regions, which
    # may appear separated in some scans
    mask[:, 0] = 0
    mask[:, -1] = 0
    mask[:, :, 0] = 0
    mask[:, :, -1] = 0
    labels = measure.label(mask, connectivity=1, background=-1)
    vals, counts = np.unique(labels, return_counts=True)
    inds = np.argsort(counts)
    # Assume that the lungs make up the third largest region
    lung_val = vals[inds][-3]
    if mask[labels == lung_val].sum() != 0:
        print('Warning: could not get mask for %s' % uid)
        mask[:] = 1
        return mask

    mask[labels == lung_val] = 1
    mask[labels != lung_val] = 0
    fill_mask(mask)
    left_center = mask[mask.shape[0] // 2, mask.shape[1] // 2, mask.shape[2] // 4]
    right_center = mask[mask.shape[0] // 2, mask.shape[1] // 2, mask.shape[2] * 3 // 4]
    if (left_center == 0) or (right_center == 0):
        print('Warning: could not get mask for %s' % uid)
        mask[:] = 1
        return mask

    mask = ndimage.morphology.binary_dilation(mask, iterations=settings.mask_dilation)
    return mask


def apply_mask(image, mask):
    image[mask == 0] = 0


def fill_mask(mask):
    for i in range(mask.shape[0]):
        slc = mask[i]
        fill_mask_slice(slc)
    for i in range(mask.shape[1]):
        slc = mask[:, i]
        fill_mask_slice(slc)
    for i in range(mask.shape[2]):
        slc = mask[:, :, i]
        fill_mask_slice(slc)


def fill_mask_slice(slc):
    labels = measure.label(slc, connectivity=1, background=-1)
    vals, counts = np.unique(labels, return_counts=True)
    inds = np.argsort(counts)
    max_val = vals[inds][-1]
    if len(vals) > 1:
        next_val = vals[inds][-2]
        labels[labels == next_val] = max_val
    slc[labels != max_val] = 1


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, uid, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    left_center = binary_image[binary_image.shape[0] // 2, binary_image.shape[1] // 2, binary_image.shape[2] // 4]
    right_center = binary_image[binary_image.shape[0] // 2, binary_image.shape[1] // 2, binary_image.shape[2] * 3 // 4]
    if (left_center == 0) or (right_center == 0):
        print('Warning: could not get mask for %s' % uid)
        binary_image[:] = 1
        return binary_image

    return binary_image