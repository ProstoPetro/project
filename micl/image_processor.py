import numpy as np
from PIL import Image
from pydicom import FileDataset
import cv2
cv2.setNumThreads(0)
import torch


class ImageProcessor:
    image_size = 336, 336
    image_mean = 0.5
    image_std = 0.2865

    def __call__(self, image: Image) -> torch.Tensor:
        """Takes a pre-processed image in PIL format
        and convert it to the torch tensor which can be fed as input to the model.

        Args:
            image (Image):
                CXR image in PIL format (grayscale).

        Returns:
            torch.Tensor:
                Tensor of size (3, H, W).
        """
        # PIL -> numpy
        image = np.array(image)
        
        if image.ndim != 2:
            raise ValueError('Image must be grayscale')
        
        if image.dtype != np.uint8:
            raise ValueError('Image must have uint8 dtype')

        # resize
        if image.shape != self.image_size:
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)

        # to (0, 1) range
        image = image.astype('float32') / 255

        # normalization
        image = (image - self.image_mean) / self.image_std

        # grayscale -> to rgb
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # HWC -> CHW
        image = np.moveaxis(image, -1, 0)

        # numpy -> torch
        image = torch.from_numpy(image)

        # add batch dim
        image = image.unsqueeze(0)

        return image

    def preprocess_dicom(self, dicom: FileDataset) -> Image:
        """Extracts an image from DICOM object,
        convert pixels values to unsigned int format, apply histogram equalization
        (as in https://physionet.org/content/mimic-cxr-jpg/2.1.0/),
        resize the image to (336, 336) resolution (we never need larger one)
        and convert it to PIL format.

        Args:
            dicom (FileDataset):
                pydicom.FileDataset object, usually obtained by pydicom.dcmread function.

        Returns:
            Image:
                The pre-processed image in PIL format.
        """
        # dicom -> numpy
        image: np.ndarray = dicom.pixel_array

        # invert pixel values if needed
        if dicom.PhotometricInterpretation == 'MONOCHROME1':
            image = image.max() - image

        # to uint8
        a, b = image.min(), image.max()
        if a == b:
            raise ValueError('The input image is all-constant (image.min() == image.max())')
        image = ((image - a) / (b - a) * 255).astype('uint8')

        # equalize histogram
        image = cv2.equalizeHist(image)

        # resize
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)

        # numpy -> PIL
        image = Image.fromarray(image)

        return image
