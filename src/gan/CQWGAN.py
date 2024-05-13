import os
import torch
import csv
from scipy.stats import wasserstein_distance_nd

from src.utils.image_utils.save_images import save_real_images, save_generated_images
from src.utils.gan_utils.gradient_penalty import compute_gp
from src.gan.nets.generator import QuantumGeneratorImported
from src.gan.nets.discriminator import Discriminator

from src.utils.gan_utils.save_value_history import save_history
from src.utils.set_seeds import set_seeds
from configs import general_configs


def train_imported_gan(train_dataloader, classes: list, out_folder: str, qasm_file_path: str,
                       metadata_file_path: str, normal_latent: bool, image_side: int = 28,
                       n_channels: int = 1, n_layers: int = 2, batch_size: int = 25,
                       n_epochs: int = 50):
    """
    Trains the generator and discriminator of the PQWGAN, using an imported circuit structure.

    :param train_dataloader: Dataloader. The dataloader containing the batches of images to be 
        used for training.
    :param classes: list. Classes of images to generate, e.g., [0,1] for MNIST.
    :param out_folder: str. The directory where the training outputs will be saved.
    :param qasm_file_path: str. file path to the QASM file to import the circuit for the generator.
    :param metadata_file_path: str. File with metadata for the circuit.
    :param normal_latent: bool. Whether to draw the latent vector from uniform or normal distribution.
    :param image_side: int. Number of pixels of the side of the image (assumed to be square).
    :param n_channels: int. Number of color channels of the image. Defaults to 1.
    :param n_layers: int. Number of layers for the quantum generator (imported ansatz is repeated
                          those many times). Defaults to 2.
    :param batch_size: int. The size of each batch of data. Defaults to 25.
    :param n_epochs: int. The number of epochs to train the model. Defaults to 50.

    :return: None.
    """
    # set_seeds(general_configs.SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    num_workers = 1 if device == 'cpu' else 8 if device == 'cuda' else 0
    print(f"Number of workers selected: {num_workers}")

    #####################
    # Setting Variables #
    #####################

    # Read the metadata from the imported circuit
    with open(metadata_file_path, mode='r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            variable_name = row['Variable'].lower().replace(' ', '_')
            if variable_name == 'n_data_qubits':
                n_data_qubits = int(row['Value'])
            elif variable_name == 'n_ancilla':
                n_ancillas = int(row['Value'])
            elif variable_name == 'patch_shape':
                patch_shape = eval(row['Value'])  # Converts string to tuple

    n_epochs = n_epochs
    classes = classes
    channels = n_channels
    image_side = image_side
    image_shape = (channels, image_side, image_side)
    ancillas = n_ancillas
    n_data_qubits = n_data_qubits
    n_tot_qubits = n_data_qubits + ancillas
    n_sub_generators = int(image_side / (int(patch_shape[0])))
    n_layers = n_layers

    # Default training parameters
    lr_D = 0.0002
    lr_G = 0.01
    b1 = 0
    b2 = 0.9
    lambda_gp = 10  # gradient penalty coefficient. Found to work well in WGAN-GP paper
    n_critic = 5  # how often to train generator compared to critic.
    sample_interval = 10  # how often to save output images

    os.makedirs(out_folder, exist_ok=True)

    print("Initializing GAN network")

    discriminator = Discriminator(img_shape=image_shape)
    generator = QuantumGeneratorImported(image_shape=image_shape,
                                         qasm_file_path=qasm_file_path,
                                         n_ancillas=n_ancillas,
                                         n_sub_generators=n_sub_generators,
                                         n_layers=n_layers)

    # Moves the networks to the selected device (CPU or GPU if available)
    discriminator = discriminator.to(device)
    generator = generator.to(device)

    # Optimizers for the generator and the discriminator
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(b1, b2))

    # Generate latent vectors. If normal_latent == True draw from normal, else draw from uniform.
    initial_latent_vector = torch.randn(batch_size, n_tot_qubits, device=device) if normal_latent \
        else torch.rand(batch_size, n_tot_qubits, device=device)

    # Save images generated by G before starting to train it.
    real_images, _ = next(iter(train_dataloader))  # Grab a batch of real images
    save_generated_images(generator=generator, latent_vector=initial_latent_vector, out_dir=out_folder,
                          image_name="0")
    # Save a sample of real images
    save_real_images(real_images=real_images, out_dir=out_folder, image_name="real_images_sample")

    #####################
    #     Training      #
    #####################

    epoch_history = []
    batch_number_history = []
    estimated_distance_history = []  # Store the Wasserstein distances
    d_loss_history = []
    g_loss_history = []
    gradient_penalty_history = []
    real_validity_history = []
    fake_validity_history = []
    emd_history = []

    file_path = os.path.join(out_folder, "training_values_history.csv")

    batches_done = 0
    g_loss = torch.tensor(0)
    # Just for the first n_critic iterations before it is calculated for the first time
    try:
        for epoch in range(n_epochs):
            print(f'Epoch number {epoch} \n')
            for i, (real_images, _) in enumerate(train_dataloader):  # Iterate over batches in the data loader.

                # Configure input
                real_images = real_images.to(device)

                #########################
                #  Train Discriminator  #
                #########################
                optimizer_D.zero_grad()  # Initialize the critic's optimizer (pytorch zero_grad).
                # generate a new z at each iteration for variability (explore the input space).
                z = torch.randn(batch_size, n_tot_qubits, device=device) if normal_latent else \
                    torch.rand(batch_size, n_tot_qubits, device=device)
                # Give generator latent vector z as input to generate images for the current iteration
                fake_images = generator(z)
                # Compute the critic's predictions for real and fake images.
                real_validity = discriminator(real_images)
                fake_validity = discriminator(fake_images)
                # Compute gradient penalty
                gradient_penalty = compute_gp(discriminator, real_images, fake_images)

                mean_real_validity = torch.mean(real_validity)
                mean_fake_validity = torch.mean(fake_validity)

                # Compute Adversarial loss - D wants to minimize it
                d_loss = -mean_real_validity + mean_fake_validity + (lambda_gp * gradient_penalty)
                # Backpropagate and update the critic's weights.
                d_loss.backward()
                optimizer_D.step()

                real_images_flat = real_images.reshape(25, -1).cpu()
                fake_images_flat = fake_images.reshape(25, -1).cpu()

                real_images_flat = real_images_flat.detach().numpy()
                fake_images_flat = fake_images_flat.detach().numpy()

                emd = wasserstein_distance_nd(real_images_flat, fake_images_flat)

                epoch_history.append(epoch)
                batch_number_history.append(i)
                real_validity_history.append(mean_real_validity.item())
                fake_validity_history.append(mean_fake_validity.item())
                d_loss_history.append(d_loss.item())
                gradient_penalty_history.append(gradient_penalty.item())
                # g_loss will be the same for n_critic steps cause calculated less often
                g_loss_history.append(g_loss.item())

                estimated_distance = torch.mean(real_validity) - torch.mean(fake_validity)
                estimated_distance_history.append(estimated_distance.item())
                emd_history.append(emd)

                optimizer_G.zero_grad()  # Initialize the generator's optimizer (pytorch zero_grad).

                if i % n_critic == 0:  # Train the generator every n_critic steps

                    #####################
                    #  Train Generator  #
                    #####################

                    # Generate a batch of fake images
                    fake_images = generator(z)
                    # Get discriminator's scores for each of those images
                    fake_validity = discriminator(fake_images)
                    # Calculate loss: the generator's ability to fool the discriminator
                    # i.e. the negative value of the scores given by the discriminator because
                    # the smaller the score, the more the generator fooled the discriminator.
                    # G wants to minimize g_loss, D wants to maximize it.
                    g_loss = -torch.mean(fake_validity)

                    # Backpropagate and update the generator's weights
                    g_loss.backward()
                    optimizer_G.step()

                    # Print and log the training progress
                    print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(train_dataloader)}]"
                          f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

                    # Save generated images and model states less often than you train the
                    # generator (else it's a bit of an overkill of saving)
                    if i % (n_critic*4) == 0 and batches_done % sample_interval == 0:
                        # Save values history
                        data = {
                            'epoch_n': epoch_history,
                            'batch_n': batch_number_history,
                            'real_validity': real_validity_history,
                            'fake_validity': fake_validity_history,
                            'd_loss': d_loss_history,
                            'gradient_penalty': gradient_penalty_history,
                            'g_loss': g_loss_history,
                            'estimated_distance': estimated_distance_history,
                            'real_emd_value': emd_history
                        }

                        save_history(data=data, file_path=file_path)

                        save_generated_images(generator=generator,
                                              latent_vector=initial_latent_vector,
                                              out_dir=out_folder,
                                              image_name=f'{batches_done}')

                        torch.save(discriminator.state_dict(),
                                   os.path.join(out_folder, 'critic-{}.pt'.format(batches_done)))
                        torch.save(generator.state_dict(),
                                   os.path.join(out_folder, 'generator-{}.pt'.format(batches_done)))
                        print("Saved images and state")

                    # Update the total number of batches done
                    batches_done += n_critic
    except Exception as e:
        print(f"An error occurred: {e}")
