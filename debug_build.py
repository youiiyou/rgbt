from types import SimpleNamespace
from datasets.build import build_dataloader

args = SimpleNamespace(
    dataset_name='VCM',
    root_dir='/data/ydl/datasets',
    num_workers=0,
    training=True,
    MLM=False,
    img_size=(384, 128),
    img_aug=False,
    text_length=77,
    sampler='random',
    batch_size=4,
    num_instance=4,
    distributed=False,
    val_dataset='test',
    test_batch_size=4,
    num_frames = 6,
)

train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)

print("num_classes:", num_classes)
print("train batches:", len(train_loader))
print("val_img batches:", len(val_img_loader))
print("val_txt batches:", len(val_txt_loader))

train_batch = next(iter(train_loader))
print("train batch keys:", train_batch.keys())
for k, v in train_batch.items():
    print(k, type(v), getattr(v, "shape", None))