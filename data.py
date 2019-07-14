from dataset import DatasetFromFolderTest, DatasetFromFolder

def get_training_set(data_dir, label_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame):
    return DatasetFromFolder(data_dir, label_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size,future_frame)


def get_eval_set(data_dir, label_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame):
    return DatasetFromFolder(data_dir, label_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size,future_frame)

def get_test_set(data_dir, label_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame):
    return DatasetFromFolderTest(data_dir, label_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame)
