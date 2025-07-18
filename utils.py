from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import numpy as np
import torch

class BinBySizeDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset wrapper that groups (bins) dataset samples by the shape of their tensor input.

    This class is useful for datasets with variable-sized tensors, such as images or sequences, allowing
    grouping of items with the same shape. Within each bin, samples are duplicated to match a fixed size,
    and bins that do not contain a sufficient number of unique items are discarded.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The original dataset. Each item is expected to be a tuple, where the first element is a tensor.
        For example, dataset[i] should return (tensor, ...).

    items_per_bin : int, optional (default=32)
        The number of items required per bin. Bins with fewer items will be oversampled
        (with repetition) to reach this count or a multiple of it.

    min_unique_items_in_bin : int, optional (default=8)
        The minimum number of unique items that must be present in a bin.
        Bins that do not meet this criterion are discarded.

    Attributes
    ----------
    dataset : torch.utils.data.Dataset
        The original dataset.

    _bins : dict
        Dictionary mapping tensor shapes to lists of dataset indices grouped by those shapes.

    _bins_keys : list
        List of bin keys (tensor shapes) for quick access.

    bins_lengths : list
        Number of items in each bin.

    bins_cumsum : numpy.ndarray
        Cumulative sum of bin lengths, used for efficient indexing.

    _items_per_bin : int
        Number of items that each bin must contain.

    Methods
    -------
    __len__()
        Returns the total number of items in the binned dataset.

    __getitem__(index)
        Returns the sample at the specified index by locating the corresponding bin and dataset index.
    """
    def __init__(self,dataset,items_per_bin = 32,min_unique_items_in_bin = 8,max_workers=8):
        """
        Bins dataset items by tensor sizes of dataset elements
        
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The original dataset. Each item is expected to be a tuple, where the first element is a tensor.
            For example, dataset[i] should return (tensor, ...).

        items_per_bin : int, optional (default=32)
            The number of items required per bin. Bins with fewer items will be oversampled
            (with repetition) to reach this count or a multiple of it.

        min_unique_items_in_bin : int, optional (default=8)
            The minimum number of unique items that must be present in a bin.
            Bins that do not meet this criterion are discarded.

        """
        super().__init__()
        bins = {}
        def get_shape(i):
            d = dataset[i]
            shapes = [item.shape for item in d if isinstance(item, torch.Tensor)]
            return str(shapes), i
        
        bins = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(get_shape, i) for i in range(len(dataset))]
            for future in as_completed(futures):
                shape_str, idx = future.result()
                bins.setdefault(shape_str, []).append(idx)


        for b in list(bins.keys()):
            if len(set(bins[b]))<min_unique_items_in_bin:
                bins.pop(b)
        
        for b in list(bins.keys()):
            bin_content = bins[b]
            size = len(set(bin_content))
            if size<min_unique_items_in_bin:
                bins.pop(b)
                continue
            while len(bin_content)<items_per_bin:
                bin_content.extend(bin_content)
                bin_content=bin_content[:items_per_bin]
                bins[b]=bin_content
            
            if len(bin_content)%items_per_bin!=0:
                expected_size = math.ceil(len(bin_content)/items_per_bin) * items_per_bin
                bin_content.extend(bin_content)
                bin_content=bin_content[:expected_size]
                bins[b]=bin_content
        
        unique_items_count = 0
        items_count = 0
        for b in bins:
            items_count+=len(bins[b])
            unique_items_count+=len(set(bins[b]))
            # print(b,len(bins[b]))
        print("unique",unique_items_count)
        print("total",items_count)
        
        bins_keys = list(bins.keys())
        bins_lengths = [len(bins[k]) for k in bins_keys]
        self.bins_lengths=bins_lengths
        self.bins_cumsum = np.cumsum(bins_lengths)
        
        self._bins=bins
        self._bins_keys=list(self._bins.keys())
        self._items_per_bin=items_per_bin
        self.dataset=dataset


    
    def __len__(self): 
        return self.bins_cumsum[-1]
    
    def __getitem__(self, index):
        bin_ind = np.where(self.bins_cumsum>index)[0][0]
        bin_size = self.bins_lengths[bin_ind]
        dataset_ind = self._bins[self._bins_keys[bin_ind]][index%bin_size]
        return self.dataset[dataset_ind]


import math
import torch
import PIL.Image
import torch.nn.functional as torch_F
class ResizeToMultiple:
    """
    Resizes input 2d tensor to have shape as multiple of `multiple_of`
    """
    def __init__(self, multiple_of: int, interpolation="nearest"):
        if multiple_of <= 0:
            raise ValueError("multiple_of must be a positive integer.")
        if interpolation not in ("bilinear", "nearest"):
            raise ValueError("Only 'bilinear' and 'nearest' interpolation are supported.")
        self.multiple_of = multiple_of
        self.interpolation = interpolation

    def __call__(self, image):
        """
        Args:
            image (PIL.Image.Image or torch.Tensor): Image to be resized.

        Returns:
            Resized image such that width and height are multiples of `multiple_of`.
        """
        if isinstance(image, PIL.Image.Image):
            width, height = image.size
            new_width = math.ceil(width / self.multiple_of) * self.multiple_of
            new_height = math.ceil(height / self.multiple_of) * self.multiple_of
            return image.resize((new_width, new_height), resample=self._pil_interpolation())

        elif isinstance(image, torch.Tensor):
            if image.ndim != 3:
                raise ValueError("Expected a 3D tensor with shape [C, H, W].")
            c, h, w = image.shape
            new_w = math.ceil(w / self.multiple_of) * self.multiple_of
            new_h = math.ceil(h / self.multiple_of) * self.multiple_of
            image = image.unsqueeze(0)  # [1, C, H, W]
            image = torch_F.interpolate(image, size=(new_h, new_w), mode=self.interpolation, align_corners=False if self.interpolation == 'bilinear' else None)
            return image.squeeze(0)
        else:
            raise TypeError("Input must be a PIL.Image.Image or a 3D torch.Tensor [C, H, W]")

    def _pil_interpolation(self):
        return {
            "bilinear": PIL.Image.BILINEAR,
            "nearest": PIL.Image.NEAREST
        }[self.interpolation]

    def __repr__(self):
        return f"{self.__class__.__name__}(multiple_of={self.multiple_of}, interpolation='{self.interpolation}')"

