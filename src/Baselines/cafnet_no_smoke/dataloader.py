from typing import Optional

from torch.utils.data import DataLoader

from collate_fn_helpers import make_rice_collate_fn
from rice_dataset import RiceDataset


def _build_dataset(
    args,
    split: str,
    base_dir: Optional[str] = None,
    split_json_path: Optional[str] = None,
) -> RiceDataset:
    return RiceDataset(
        base_dir=base_dir or args.base_dir,
        split_json_path=args.split_json if split_json_path is None else split_json_path,
        split=split,
        input_height=args.input_height,
        input_width=args.input_width,
        patch_size=args.patch_size,
    )


def _build_loader(
    args,
    split: str,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    pin_memory: bool,
    base_dir: Optional[str] = None,
    split_json_path: Optional[str] = None,
):
    dataset = _build_dataset(
        args,
        split=split,
        base_dir=base_dir,
        split_json_path=split_json_path,
    )
    collate_fn = make_rice_collate_fn(
        input_height=args.input_height,
        input_width=args.input_width,
        radar_max_depth_m=args.radar_max_depth_m,
        max_dist_correspondence=args.max_dist_correspondence,
        patch_size=dataset.patch_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def create_train_test_loaders(args, pin_memory: bool = False):
    train_loader = _build_loader(
        args,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
    )
    test_loader = _build_loader(
        args,
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def create_inference_loader(args, pin_memory: bool = False):
    """Create the single packaged Smoke-Eval loader used for inference."""

    test_base_dir = getattr(args, "test_base_dir", "")
    if not test_base_dir:
        raise ValueError("Config must define 'test_base_dir' for inference.")

    test_split = getattr(args, "test_split", "train")
    test_split_json = getattr(args, "test_split_json", None)
    if not test_split_json:
        test_split_json = None

    return _build_loader(
        args,
        split=test_split,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        base_dir=test_base_dir,
        split_json_path=test_split_json,
    )
