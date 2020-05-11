import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from public.reid.utils import CoverageKPSampler, VehicleTrainDataset, VehicleValDataset, RandomShrink, RandomErasing


class Config(object):
    network_name = "res50"
    log = './log'  # Path to save log
    checkpoints = './checkpoints'
    resume = './checkpoints/latest.pth'
    evaluate = None  # evaluate model path
    root_path = '/data/AICITY2020/AIC20_track2/'
    train_dataset_pkl = [
        '/data/aicity_pkl/track2_train_pytorch.pkl',
        '/data/aicity_pkl/track2_simu_train_pytorch.pkl',
    ]
    val_dataset_pkl = '/data/aicity_pkl/benchmark_pytorch.pkl'

    pretrained = True
    num_classes = 1695
    seed = 0
    input_image_size = 320

    train_dataset = VehicleTrainDataset(
        root_path,
        train_dataset_pkl,
        input_size=(input_image_size, input_image_size),
        use_crop_box=True,
        use_background_substitution=False,
        transform=transforms.Compose([
            RandomShrink(),
            transforms.ToPILImage(),
            transforms.Resize((input_image_size, input_image_size)),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0)
            ],
                                   p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.Pad(10),
            transforms.RandomCrop(input_image_size),
            RandomErasing(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]))

    val_dataset = VehicleValDataset(root_path,
                                    val_dataset_pkl,
                                    input_size=(320, 320),
                                    use_crop_box=True,
                                    transform=transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((320, 320)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                                    ]))

    P = 64
    K = 4
    batch_size = int(P * K)
    num_workers = 8

    sampler = CoverageKPSampler(train_dataset.metas, P * K, K)

    train_loader = DataLoader(train_dataset,
                              sampler=sampler,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True)

    milestones = [40, 70]
    warm_up_epochs = 10
    epochs = 120
    accumulation_steps = 1
    lr = 3.5e-4
    weight_decay = 5e-4
    print_interval = 100


if __name__ == '__main__':
    config = Config()
