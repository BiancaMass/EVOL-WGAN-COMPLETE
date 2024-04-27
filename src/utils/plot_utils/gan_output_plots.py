import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def plot_gan_outputs(input_csv_file, output_dir):

    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(input_csv_file)

    # Calculate the iteration for the x-axis
    df['batch_n'] += 1
    df['iteration'] = df['epoch_n'] * max(df['batch_n']) + df['batch_n']

    # Set seaborn theme for all plots
    sns.set_theme()

    # For the plot titles
    titles = {
        'real_validity': 'Critic score for real images',
        'fake_validity': 'Critic score for generated images',
        'd_loss': 'Discriminator Loss',
        'gradient_penalty': 'Gradient Penalty',
        'g_loss': 'Generator Loss',
        'estimated_distance': 'Estimated Wasserstein Distance',
        'emd_history': 'Calculated Wasserstein Distance'
    }

    # For the y axis labels
    labels = {
        'real_validity': 'score',
        'fake_validity': 'score',
        'd_loss': 'loss',
        'gradient_penalty': 'gradient penalty',
        'g_loss': 'loss',
        'estimated_distance': 'estimated distance',
        'emd_history': 'EMD'
    }

    for column in df.columns:
        if df[column].dtype not in ['int64', 'float64'] or column in ['epoch_n', 'batch_n',
                                                                      'iteration']:
            continue

        # Create a plot for each numeric column
        plt.figure(figsize=(10, 5))
        plt.plot(df['iteration'], df[column], marker='o', linestyle='-', markersize=5)
        plt.xlabel('Iteration')
        plt.ylabel(labels.get(column, column.title()))

        plt.title(titles.get(column, f'{column.title()} Values Over Iterations'))
        plt.grid(True)

        plot_filename = f'{column}_plot.png'
        # plt.show()

        plt.savefig(os.path.join(plots_dir, plot_filename))
        plt.close()

    # Additional plot for real_validity and fake_validity
    plt.figure(figsize=(10, 5))
    plt.plot(df['iteration'], df['real_validity'], marker='o', linestyle='-', color='blue',
             label='Real images')
    plt.plot(df['iteration'], df['fake_validity'], marker='x', linestyle='-', color='red',
             label='Generated images')
    plt.xlabel('Iteration')
    plt.ylabel('Critic Score')
    plt.title('Critic Score for real and generated images over iterations')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(os.path.join(plots_dir, 'real_and_fake_validity_plot.png'))
    plt.close()


