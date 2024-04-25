import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_best_fitness(csv_file, destination_path):
    df = pd.read_csv(csv_file)
    emd_values = df['Best Fitness'].dropna()

    sns.set_theme()
    plt.figure(figsize=(10, 5))
    plt.plot(emd_values, marker='o', linestyle='-', markersize=5)
    plt.xlabel('Generation')
    plt.ylabel('EMD Score')

    # Set x-axis ticks every 10 generations
    # Ensure we are not attempting to create more ticks than there are generations
    num_ticks = len(emd_values) // 10
    plt.xticks(range(0, len(emd_values), 10 if num_ticks > 0 else 1))

    plt.title('EMD values over generations')
    plt.grid(True)

    # Save the plot to the specified path
    plt.savefig(destination_path)
    plt.close()
