import torch

from src.utils.gan_utils.fid_score_fun import compute_fid
from src.gan.nets.generator import QuantumGeneratorImported
import configs.general_configs as general_configs
import configs.config_gan as gan_config


def load_real_images(dataloader, num_images):
    real_images = []
    for i, (images, _) in enumerate(dataloader):
        real_images.append(images)
        if len(real_images) * dataloader.batch_size >= num_images:
            break
    return torch.cat(real_images)[:num_images]


def fid_score_calculator(validation_dataloader, n_images_to_evaluate, path_to_last_generator, qasm_file_path):
    device = 'cpu'
    num_workers = 0 if device == 'cpu' else 8 if device == 'cuda' else 0
    print(f"Number of workers selected: {num_workers}")

    image_side = general_configs.IMAGE_SIDE
    channels = general_configs.N_CHANNELS
    classes = general_configs.CLASSES
    n_tot_qubits = general_configs.N_DATA_QUBITS + general_configs.N_ANCILLAS
    patch_shape = (general_configs.PATCH_HEIGHT, general_configs.PATCH_WIDTH)
    n_sub_generators = int(image_side / (int(patch_shape[0])))
    n_layers = gan_config.N_LAYERS

    randn_true = gan_config.RANDN

    # evol_batch_size = es_configs.EVOL_BATCH_SIZE
    # gan_batch_size = gan_config.GAN_BATCH_SIZE
    # evol_train_size = general_configs.EVOL_TRAIN_SIZE
    # evol_val_size = general_configs.EVOL_VALID_SIZE
    # gan_train_size = general_configs.GAN_TRAIN_SIZE
    # gan_val_size = general_configs.GAN_VALID_SIZE

    # validation_dataloader, _, _, _ = dataloader_mnist(
    #     num_workers=num_workers,
    #     data_dir=general_configs.DATASET_DIR,
    #     image_side=image_side,
    #     classes=classes,
    #     train=True,
    #     evol_batch_size=evol_batch_size,
    #     gan_batch_size=gan_batch_size,
    #     gan_train_size=n_images_to_evaluate,
    #     gan_val_size=gan_val_size,
    #     evol_train_size=evol_train_size,
    #     evol_val_size=evol_val_size)

    # Load the generator
    generator = QuantumGeneratorImported(image_shape=(channels, image_side, image_side),
                                         qasm_file_path=qasm_file_path,
                                         n_ancillas=general_configs.N_ANCILLAS,
                                         n_sub_generators=n_sub_generators,
                                         n_layers=n_layers)
    generator = generator.to(device)
    generator.load_state_dict(torch.load(path_to_last_generator))

    # Generate fake images
    z = torch.randn(n_images_to_evaluate, n_tot_qubits, device=device) if randn_true else \
        torch.rand(n_images_to_evaluate, n_tot_qubits, device=device)
    fake_images = generator(z)

    # Load real images
    dataloader = validation_dataloader
    real_images = load_real_images(dataloader, n_images_to_evaluate)

    fid_score = compute_fid(real_images, fake_images)
    return fid_score


# if __name__ == '__main__':
#     fid_score_calculator(n_images_to_evaluate=500,
#                          path_to_last_generator='/Users/bmassacci/main_folder/maastricht/academics/quantum_thesis/scripts/EVOL-WGAN-COMPLETE/output/24_04_28_12_56_02/gan/generator-20.pt',
#                          qasm_file_path='/Users/bmassacci/main_folder/maastricht/academics/quantum_thesis/scripts/EVOL-WGAN-COMPLETE/output/24_04_28_12_56_02/evol/final_best_circuit.qasm')
