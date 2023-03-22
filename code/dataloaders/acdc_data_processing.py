import glob
import h5py
import numpy as np
import os
import pdb
import SimpleITK as sitk

slice_num = 0
mask_path = sorted(glob.glob("/mntnfs/med_data5/yezhu/training/patient*/*_frame[0-9][0-9].nii.gz"))
for case in mask_path:
    img_itk = sitk.ReadImage(case)
    origin = img_itk.GetOrigin()
    spacing = img_itk.GetSpacing()
    direction = img_itk.GetDirection()
    image = sitk.GetArrayFromImage(img_itk)
    msk_path = case.replace('.nii', "_gt.nii")
    if os.path.exists(msk_path):
        print(msk_path)
        item = case.split("/")[-1].split(".")[0]
        msk_itk = sitk.ReadImage(msk_path)
        mask = sitk.GetArrayFromImage(msk_itk)

        # save cases
        f = h5py.File(
                '/mntnfs/med_data5/yezhu/datasets/ACDC/volumes/{}.h5'.format(item), 'w')
        f.create_dataset(
                'image', data=image, compression="gzip")
        f.create_dataset('mask', data=mask, compression="gzip")
        f.close()


        image = (image - image.min()) / (image.max() - image.min())
        print(image.shape)
        image = image.astype(np.float32)
        if image.shape != mask.shape:
            print("Error")
        print(item)
        for slice_ind in range(image.shape[0]):
            f = h5py.File(
                '/mntnfs/med_data5/yezhu/datasets/ACDC/slices/{}_slice_{}.h5'.format(item, slice_ind), 'w')
            f.create_dataset(
                'image', data=image[slice_ind], compression="gzip")
            f.create_dataset('mask', data=mask[slice_ind], compression="gzip")
            f.close()
            slice_num += 1
print("Converted all ACDC volumes to 2D slices")
print("Total {} slices".format(slice_num))
