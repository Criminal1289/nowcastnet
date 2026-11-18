from nowcasting.data_provider.bj_loader import BJDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np

def data_provider(configs, split='test'):
 
    if configs.dataset_name == 'radar':
        from nowcasting.data_provider import loader
        test_input_param = {
                            'image_width': configs.img_width,
                            'image_height': configs.img_height,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': configs.dataset_name + 'test iterator',
                            'total_length': configs.total_length,
                            'data_path': configs.dataset_path,
                            'type': 'test',
                            }
        test_input_handle = loader.InputHandle(test_input_param)
        test_input_handle = DataLoader(test_input_handle,
                                       batch_size=configs.batch_size,
                                       shuffle=False,
                                       num_workers=configs.cpu_worker,
                                       drop_last=True)

    elif configs.dataset_name == 'bj':
        is_train = (split == 'train')
        stride = getattr(configs, 'stride_steps', 1) if is_train else getattr(configs, 'eval_stride_steps', 1)
        dataset = BJDataset(
            data_root=configs.dataset_path,
            split=split,
            input_length=getattr(configs, 'input_length', 11),
            pred_length=getattr(configs, 'pred_length', 10),
            stride_steps=stride,
            allow_interpolate=getattr(configs, 'allow_interpolate', None),
            patch_size=getattr(configs, 'patch_size', 0) if is_train else 0,
            pad_to=getattr(configs, 'pad_to', 320),
            filter_dry=getattr(configs, 'filter_dry', False) if is_train else False,
            dry_keep_prob=getattr(configs, 'dry_keep_prob', 0.2),
            rain_threshold=getattr(configs, 'rain_threshold', 0.1),
        )
        test_input_handle = DataLoader(
            dataset,
            batch_size=configs.batch_size if is_train else getattr(configs, 'eval_batch_size', 1),
            shuffle=is_train,
            num_workers=configs.cpu_worker,
            drop_last=is_train,
            pin_memory=True,
        )

    else:
        raise ValueError('Name of dataset unknown %s' % configs.dataset_name)

    return test_input_handle

