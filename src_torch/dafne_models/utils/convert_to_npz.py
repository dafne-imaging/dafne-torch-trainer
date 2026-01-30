import os
import SimpleITK as sitk
import numpy as np

# convert panters dataset into npz

def read_mha(image_path, mask_path):
    itk_img = sitk.ReadImage(image_path)
    itk_mask = sitk.ReadImage(mask_path)

    assert itk_img.GetSpacing() == itk_mask.GetSpacing()
    assert itk_img.GetDirection() == itk_mask.GetDirection()
    assert itk_img.GetOrigin() == itk_mask.GetOrigin()

    img_np = sitk.GetArrayFromImage(itk_img)   # (Z, Y, X)
    mask_np = sitk.GetArrayFromImage(itk_mask)

    assert img_np.shape == mask_np.shape, "Image and mask shape mismatch"

    return img_np, mask_np


def save_slices(image, mask, output_dir, patient_name):
    patient_folder = os.path.join(output_dir, patient_name)
    os.makedirs(patient_folder, exist_ok=True)

    for idx in range(image.shape[0]):
        filename = f"slice_{idx:03d}.npz"
        np.savez_compressed(
            os.path.join(patient_folder, filename),
            arr_0=image[idx],
            arr_1 =mask[idx]
        )

if __name__ == "__main__":

    input_root = "Test_images/PANTHER_Task2"
    images_dir = os.path.join(input_root, "ImagesTr")
    labels_dir = os.path.join(input_root, "LabelsTr")

    output_dir = "Test_images/npz_test"
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(f for f in os.listdir(images_dir) if f.endswith(".mha"))

    for file_name in image_files:
        image_path = os.path.join(images_dir, file_name)
        mask_path = os.path.join(labels_dir, file_name.replace('_0000', ''))

        if not os.path.exists(mask_path):
            print(f"Mask missing for {file_name}, skipping")
            continue

        patient_name = os.path.splitext(file_name)[0]

        print(f"Processing {patient_name}...")

        image_3d, mask_3d = read_mha(image_path, mask_path)

        save_slices(
            image_3d,
            mask_3d,
            output_dir,
            patient_name
        )

    print("Conversion completed")