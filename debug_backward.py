import torch
from types import SimpleNamespace

from datasets.build import build_dataloader
from model import build_model


def main():
    args = SimpleNamespace(
        dataset_name='VCM',
        root_dir='/data/ydl/datasets',

        # dataloader
        num_workers=0,
        training=True,
        MLM=False,
        sampler='random',
        batch_size=4,
        test_batch_size=4,
        num_instance=4,
        distributed=False,
        val_dataset='test',

        # image / text
        img_size=(384, 128),
        img_aug=False,
        text_length=77,
        vocab_size=49408,

        # model
        pretrain_choice='ViT-B/16',
        stride_size=16,
        temperature=0.02,
        loss_names='sdm+id',
        id_loss_weight=1.0,
        mlm_loss_weight=1.0,
        cmt_depth=4,

        # misc
        local_rank=0,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device)

    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    print("num_classes:", num_classes)

    model = build_model(args, num_classes)
    model.to(device)
    model.eval()

    batch = next(iter(train_loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    print("batch keys:", batch.keys())
    for k, v in batch.items():
        print(k, v.shape, v.dtype, v.device)

    model.train()
    ret = model(batch)

    total_loss = 0.0
    for k, v in ret.items():
        if "loss" in k:
            total_loss = total_loss + v

    print("total_loss:", total_loss.item())

    model.zero_grad()
    total_loss.backward()

    print("backward ok")

    print("ret keys:", ret.keys())
    for k, v in ret.items():
        if torch.is_tensor(v):
            if v.ndim == 0:
                print(k, v.item(), v.dtype, v.device)
            else:
                print(k, v.shape, v.dtype, v.device)
        else:
            print(k, type(v), v)


if __name__ == "__main__":
    main()