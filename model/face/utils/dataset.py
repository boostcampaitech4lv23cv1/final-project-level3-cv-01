import cv2
import numpy as np
from PIL import Image
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.folder import DatasetFolder, default_loader, IMG_EXTENSIONS
from facial_analysis import FacialImageProcessing


imgProcessing = FacialImageProcessing()

TARGET_SIZE = (224, 224)


def process_face_image(img, path):
    try:
        img = np.array(img)
        bounding_boxes, points = imgProcessing.detect_faces(img)
        best_bb = []
        best_square = 0
        for b in bounding_boxes:
            b = [int(bi) for bi in b]
            x1, y1, x2, y2 = b[0:4]
            if x2 > x1 and y2 > y1:
                sq = (x2 - x1) * (y2 - y1)
                if sq > best_square:
                    best_square = sq
                    best_bb = b

        if len(best_bb) != 0:
            img_h, img_w, _ = img.shape
            face_x, face_y = best_bb[0], best_bb[1]
            face_w, face_h = (best_bb[2] - best_bb[0]), (best_bb[3] - best_bb[1])
            dw, dh = 20, 40

            box = (
                max(0, face_x - dw),
                max(0, face_y - dh),
                min(img_w, face_x + face_w + dw),
                min(img_h, face_y + face_h + dh),
            )

            face_img = img[box[1] : box[3], box[0] : box[2], :]
            face_img = cv2.resize(face_img, TARGET_SIZE)

            return Image.fromarray(face_img)
        else:
            # print("No faces found for ", path)
            return Image.fromarray(img)

    except IOError:
        print("cannot create facial image for '%s'" % path)


class customDataset(DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = process_face_image(sample, path)
        # print(type(sample))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
