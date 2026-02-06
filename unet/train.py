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

if __name__ == '__main__':
    input_path = Path('P3M-500-NP/original_image')
    label_path = Path('P3M-500-NP/mask')
    BATCH_SIZE = 16

    writer = SummaryWriter(log_dir='runs/unet_experiment_1')

    train_transform = v2.Compose([
        v2.Resize((1024, 1024)),
        v2.RandomCrop((768, 768)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=10),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = v2.Compose([
        v2.Resize((1024, 1024)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



    full_dataset = P3MDataset(
        input_path=input_path,
        label_path=label_path
    )

    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=4,
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
    loss_fn = BceDiceLoss
    optimizer = optim.RMSprop(params=model.parameters(), lr=1e-4, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, 
        mode='min',
        patience=2, 
    )

    epoch_tqdm = tqdm(range(EPOCH), unit='batch')
    global_step = 0
    for epoch in epoch_tqdm:
        epoch_loss = 0
        epoch_tqdm.set_description(f'epoch {epoch}/{EPOCH}')
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(inputs)
            loss = loss_fn(inputs, preds)

            loss.backward()
            optimizer.step()
            optimizer.update()
            epoch_tqdm.set_postfix(**{'loss (batch)': loss.item()})
            global_step += 1

            writer.add_scalar('Loss/train', loss.item(), global_step=global_step)
                
