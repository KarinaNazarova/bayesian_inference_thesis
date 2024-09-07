import pandas as pd
from copulas.multivariate import GaussianMultivariate
from copulas.multivariate import VineCopula

def generate_synthetic_data(input_file, output_file):
    # load data 
    df = pd.read_excel(input_file, index_col=None, header=0)

    # fit a Copula Model
    copula = VineCopula(vine_type='center')
    copula.fit(df)

    # generate Synthetic Data
    synthetic_data = copula.sample(len(df))

    # compute means for original and synthetic data
    original_means = df.mean()
    synthetic_means = synthetic_data.mean()

    # compare means 
    mean_comparison = pd.DataFrame({
        'Original Mean': original_means,
        'Synthetic Mean': synthetic_means
    })

    print("Mean Comparison:")
    print(mean_comparison)

    synthetic_data.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")