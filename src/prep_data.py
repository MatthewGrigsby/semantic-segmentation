from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import cv2
from PIL import Image
from patchify import patchify


class PrepImages:
    def __init__(self,
                 image_directory_path,
                 mask_directory_path,
                 root_directory_path,
                 patch_size,
                 overlap
                 ):
        self.image_directory_path = image_directory_path
        self.mask_directory_path = mask_directory_path
        self.root_directory_path = root_directory_path
        self.patch_size = patch_size
        self.overlap = overlap
        self.n_classes = None
        self.classes = None

    def create_patches(self, overlap=0.5):
        if overlap > 0:
            self.create_overlap_tiles(
                path_to_img=self.image_directory_path,
                root_directory=self.root_directory_path,
                patch_size=self.patch_size,
                overlap=self.overlap,
                images_or_masks='images'
            )
            self.create_overlap_tiles(
                path_to_img=self.mask_directory_path,
                root_directory=self.root_directory_path,
                patch_size=self.patch_size,
                overlap=self.overlap,
                images_or_masks='masks'
            )
        else:
            self.crop_and_save(self.image_directory_path, self.root_directory_path, self.patch_size, 'images')
            self.crop_and_save(self.mask_directory_path, self.root_directory_path, self.patch_size, 'masks')

    def remove_1class_images(self):
        useless = 0
        train_img_dir = self.root_directory_path + 'n' + str(self.patch_size) + "_patches/images/"
        train_msk_dir = self.root_directory_path + 'n' + str(self.patch_size) + "_patches/masks/"
        img_list = os.listdir(train_img_dir)
        msk_list = os.listdir(train_msk_dir)

        new_root = self.root_directory_path + 'n' + str(self.patch_size) + '_patches/images_with_useful_info'
        new_img_dir = new_root + '/images/'
        new_msk_dir = new_root + '/masks/'
        if not os.path.exists(new_root):
            os.mkdir(new_root)
            os.mkdir(new_img_dir)
            os.mkdir(new_msk_dir)
            print(new_img_dir)
            print(new_msk_dir)

        for img in range(len(img_list)):
            img_name = img_list[img]
            mask_name = msk_list[img]

            temp_image = cv2.imread(train_img_dir + img_list[img], 1)
            temp_mask = cv2.imread(train_msk_dir + msk_list[img], 0)
            val, counts = np.unique(temp_mask, return_counts=True)
            if (1 - (counts[0] / counts.sum())) > 0.05:
                cv2.imwrite(new_img_dir + img_name, temp_image)
                cv2.imwrite(new_msk_dir + mask_name, temp_mask)
            else:
                useless += 1
        print("Total useful images are: ", len(img_list) - useless)
        print("Total less-useful images are: ", useless)

        def encode_mask_classes(self, train_masks):
            label_encoder = LabelEncoder()
            n, h, w = train_masks.shape
            train_masks_reshaped = train_masks.reshape(-1, 1)
            train_masks_reshaped_encoded = label_encoder.fit_transform(train_masks_reshaped)
            train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
            self.classes = np.unique(train_masks_encoded_original_shape)
            self.n_classes = len(self.classes)

            print(f"Classes: {str(self.classes)}")
            return train_masks_reshaped_encoded

    @staticmethod
    def create_overlap_tiles(path_to_img, root_directory, patch_size, overlap, images_or_masks, frmt='.tif'):
        def start_points(size, split_size, overlap=0):
            points = [0]
            stride = int(split_size * (1 - overlap))
            counter = 1
            while True:
                pt = stride * counter
                if pt + split_size >= size:
                    points.append(size - split_size)
                    break
                else:
                    points.append(pt)
                counter += 1
            return points
        for path, sub_dirs, files in os.walk(path_to_img):
            # print(dir_name)
            images = os.listdir(path)
            for i, image_name in enumerate(images):
                if image_name.endswith(".tif"):
                    file_path = path + "/" + image_name
                    print(file_path)
                    if images_or_masks == "images":
                        img = cv2.imread(file_path, 1)
                    else:
                        img = cv2.imread(file_path, 0)
                    img_h, img_w = img.shape[0], img.shape[1]
                    split_width = patch_size
                    split_height = patch_size
                    X_points = start_points(img_w, split_width, overlap)
                    Y_points = start_points(img_h, split_height, overlap)

                    count = 0
                    for j in Y_points:
                        for k in X_points:
                            save_dir = root_directory + 'n' + str(patch_size) + "_patches/" + images_or_masks + "/" + \
                                       image_name + "patch_" + str(j) + '_' + str(k) + frmt
                            split_img = img[j:j + split_height, k:k + split_width]
                            cv2.imwrite(save_dir, split_img)
                            count += 1
                    print(f"Saved {count} {images_or_masks} patches of {patch_size} x {patch_size} dimensions at: "
                          f"{root_directory + 'n' + str(patch_size) + '_patches/' + images_or_masks}")


    @staticmethod
    def overlap_tile_processing(img_array, expend_px_width, expend_px_height):
        """
        Following U-Net paper 'Overlap-tile strategy' processing image
        :param img_array: input image array
        :param expend_px_width: per edge expend width ex. 512*512 => 512*(512+(92*2))
        :param expend_px_height: per edge expend height ex. 512*512 => (512+(92*2))*512
        :return: processed image array
        """
        import cv2

        def flip_horizontally(np_array):
            return cv2.flip(np_array, 1)

        def flip_vertically(np_array):
            return cv2.flip(np_array, 0)

        original_height = img_array.shape[0]
        original_width = img_array.shape[1]

        # Expand width first
        # left:
        left_result = flip_horizontally(img_array[0:0 + original_height, 0:0 + expend_px_width])
        # right:
        right_result = flip_horizontally(
            img_array[0:0 + original_height, original_width - expend_px_width: original_width])

        result_img = cv2.hconcat([left_result, img_array])
        result_img = cv2.hconcat([result_img, right_result])

        result_img_height = result_img.shape[0]
        result_img_width = result_img.shape[1]

        # Expand height
        top_result = flip_vertically(result_img[0:0 + expend_px_height, 0:0 + result_img_width])
        bottom_result = flip_vertically(
            result_img[result_img_height - expend_px_height: result_img_height, 0:0 + result_img_width])

        result_img = cv2.vconcat([top_result, result_img])
        result_img = cv2.vconcat([result_img, bottom_result])

        return result_img


    @staticmethod
    def crop_and_save(directory, root_directory, patch_size, image_or_mask):
        # print('starting')
        for path, sub_dirs, files in os.walk(directory):
            dir_name = path.split(os.path.sep)[-1]
            # print(dir_name)
            images = os.listdir(path)
            for j, image_name in enumerate(images):
                # print(j)
                # print(image_name)
                if image_name.endswith(".tif"):
                    file_path = path + "/" + image_name
                    print(file_path)
                    if image_or_mask == "images":
                        image = cv2.imread(file_path, 1)
                    else:
                        image = cv2.imread(file_path, 0)
                    SIZE_X = (image.shape[1] // patch_size) * patch_size
                    SIZE_Y = (image.shape[0] // patch_size) * patch_size
                    image = Image.fromarray(image)
                    image = image.crop((0, 0, SIZE_X, SIZE_Y))
                    image = np.array(image)
                    print('Cropped image size: ', image.shape)
                    print("Patching:", path + "/" + image_name)
                    if image_or_mask == "images":
                        patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
                    else:
                        patches_img = patchify(image, (patch_size, patch_size), step=patch_size)
                    # print(patches_img.shape)
                    for j in range(patches_img.shape[0]):
                        for k in range(patches_img.shape[1]):
                            single_patch_img = patches_img[j, k, :, :]
                            # print(single_patch_img.shape)
                            if image_or_mask == "images":
                                single_patch_img = single_patch_img[0]
                            # print(single_patch_img.shape)
                            save_dir = root_directory + 'n' + str(
                                patch_size) + "_patches/" + image_or_mask + "/" + image_name + \
                                       "patch_" + str(j) + str(
                                k) + ".tif"
                            # print(save_dir)
                            cv2.imwrite(save_dir, single_patch_img)


def start_points(size, split_size, overlap=0):
            points = [0]
            stride = int(split_size * (1 - overlap))
            counter = 1
            while True:
                pt = stride * counter
                if pt + split_size >= size:
                    points.append(size - split_size)
                    break
                else:
                    points.append(pt)
                counter += 1
            return points