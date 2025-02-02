import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def main():
    # Define the directory containing the CSV files
    # directory = "results"
    main_directory = "results"

    # List to hold DataFrames
    df_list = []

    # Walk through all subdirectories and files
    for root, _, files in os.walk(main_directory):
        for file in files:
            if file.endswith('results.csv'):
                file_path = os.path.join(root, file)
                # Extract subdirectory name (relative to main directory)
                subdirectory_name = os.path.relpath(root, main_directory)

                # Read the CSV file
                df = pd.read_csv(file_path,index_col=0)

                # Add the subdirectory name as a new column
                df['dynamics'] = subdirectory_name

                # Append the DataFrame to the list
                df_list.append(df)

    # Concatenate all collected DataFrames
    combined_df = pd.concat(df_list, ignore_index=True)

    # Display the combined DataFrame
    # print(combined_df.head())

    # Optionally, save the combined DataFrame to a new CSV file
    combined_df.to_csv("collated_results.csv", index=False)

    # print(combined_df['dynamics'].unique())

    df = combined_df

    # print(df)
    df = df[df.output_size == 7]

    fig,ax = plt.subplots()
    sns.stripplot(
        x='dynamics',
        y='test_losses_mean',
        hue='num_layers',
        data=df,
        ax=ax,
    )
    ax.set_yscale('log')
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig('test_plots/benchmarks_num_layers.png',dpi=300)



    df = combined_df
    df = df[df.num_layers==6]
    fig,ax = plt.subplots()
    sns.stripplot(
        x='dynamics',
        y='test_losses_mean',
        hue='output_size',
        data=df,
        ax=ax,
    )
    ax.set_yscale('log')
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig('test_plots/benchmarks_output_size.png',dpi=300)

if __name__ == '__main__':
    main()