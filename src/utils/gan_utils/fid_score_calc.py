import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.utils.image_utils.data_loader_MNIST import dataloader_mnist
from src.utils.gan_utils.fid_score_fun import compute_fid

from src.gan.nets.generator import QuantumGeneratorImported

import configs.general_configs as general_configs
import configs.config_evol as es_configs
import configs.config_gan as gan_config



def load_real_images(dataloader, num_images):
    real_images = []
    for i, (images, _) in enumerate(dataloader):
        real_images.append(images)
        if len(real_images) * dataloader.batch_size >= num_images:
            break
    return torch.cat(real_images)[:num_images]


def generate_fake_images(generator, num_images, z_dim, device):
    z = torch.randn(num_images, z_dim, device=device)
    fake_images = generator(z)
    return fake_images


def main():
    # Set device
    device = 'cpu'
    num_workers = 0 if device == 'cpu' else 8 if device == 'cuda' else 0
    print(f"Number of workers selected: {num_workers}")

    image_side = general_configs.IMAGE_SIDE
    classes = general_configs.CLASSES
    evol_batch_size = es_configs.EVOL_BATCH_SIZE
    gan_batch_size = gan_config.GAN_BATCH_SIZE
    evol_train_size = general_configs.EVOL_TRAIN_SIZE
    evol_val_size = general_configs.EVOL_VALID_SIZE
    gan_train_size = general_configs.GAN_TRAIN_SIZE
    gan_val_size = general_configs.GAN_VALID_SIZE

    train_gan_loader, _, _, _ = dataloader_mnist(
                                        num_workers=num_workers,
                                        data_dir=general_configs.DATASET_DIR,
                                        image_side=image_side,
                                        classes=classes,
                                        train=True,
                                        evol_batch_size=evol_batch_size,
                                        gan_batch_size=gan_batch_size,
                                        gan_train_size=1000,
                                        gan_val_size=gan_val_size,
                                        evol_train_size=evol_train_size,
                                        evol_val_size=evol_val_size)

    # Load the generator
    Generator = QuantumGeneratorImported(image_shape=(1, 28, 28),
                                         qasm_file_path=...,
                                         n_ancillas=general_configs.N_ANCILLAS,
                                         n_sub_generators=...,
                                         n_layers=...)
    generator = Generator().to(device)  # Ensure Generator is correctly defined or imported
    generator.load_state_dict(torch.load('path_to_your_generator.pt'))
    generator.eval()

    dataloader = train_gan_loader

    # Load real images
    real_images = load_real_images(dataloader, 1000)

    # Generate fake images
    fake_images = generate_fake_images(generator, 1000, z_dim=100, device=device)  # Adjust z_dim as needed

    # Calculate FID score
    fid_score = compute_fid(real_images, fake_images)
    print("FID Score:", fid_score)


if __name__ == '__main__':
    main()
