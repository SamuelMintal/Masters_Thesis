import pandas as pd
import os
import argparse

def main(args):
    accumulated_dfs = []

    for df_file_name in os.listdir(args.dir_path):
        df_file_path = os.path.join(args.dir_path, df_file_name)        
        accumulated_dfs.append(pd.read_csv(df_file_path))

    result_df = pd.concat(accumulated_dfs)
    result_df.reset_index(drop=True, inplace=True)

    print(f'Amount of architectures in merged CSV in {len(result_df)}')

    result_df.to_csv(os.path.join(args.dir_path, args.save_file_name), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, help='path to directory containing CSVs to merge')
    parser.add_argument('--save_file_name', type=str, default='_all_merged.csv', help='Name of the resulting merged CSV')
    args = parser.parse_args()

    main(args)