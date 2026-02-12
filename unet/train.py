from typing import Any
from numpy import random
import torch
from data import P3MDataset
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2
from pathlib import Path
from tqdm import tqdm
from loss import BceDiceLoss
import torch.optim as optim
from UNET import UNET
from torch.utils.tensorboard import SummaryWriter
from utils import get_random_test_images, un_normalize, visualize
if __name__ == '__main__':
    input_path = Path('P3M-500-NP/original_image')
    label_path = Path('P3M-500-NP/mask')
    BATCH_SIZE = 4

    writer = SummaryWriter(log_dir='runs/unet_experiment_1')

    train_transform = v2.Compose([
        v2.Resize((512, 512)),
        # v2.RandomCrop((312, 312)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=10),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # test_transform = v2.Compose([
    #     v2.Resize((512, 512)),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    full_dataset = P3MDataset(
        input_path=input_path,
        label_path=label_path,
        transform=train_transform
    )

    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE
    )

    device = 'mps'

    model = UNET(n_channels=3, n_class=1)
    model = model.to(device)


    EPOCH = 200
    loss_fn = BceDiceLoss()
    optimizer = optim.RMSprop(params=model.parameters(), lr=1e-4, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, 
        mode='min',
        factor=0.1,
        patience=2, 
    )

    global_step = 0
    for epoch in range(EPOCH):
        epoch_loss = 0
        nums_item = 0
        train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), unit='batch', desc=f"Epoch {epoch + 1}")
        for i, (inputs, labels) in train_loader_tqdm:
            model.train()
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            preds = model(inputs)
            loss = loss_fn(preds, labels)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
                
            train_loader_tqdm.set_postfix(**{'train_loss': loss.item()})
            epoch_loss += loss.item()
            nums_item += 1
            writer.add_scalar('Loss/train', loss.item(), global_step=global_step)
            if global_step > 0 and global_step % 20 == 0:
                visualize(model=model, data_loader=test_loader, device=device, num_images=4, writer=writer, step=global_step)
            global_step += 1
        avg_train_loss = epoch_loss/nums_item
        print(f"Epoch {epoch + 1}: train loss: {avg_train_loss}")
        scheduler.step(avg_train_loss)
        model.eval()


    writer.close()    

                
