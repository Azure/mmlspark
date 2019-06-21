import numpy as np
import pandas as pd
from sys import argv
import os


def main(args):
    user_parquet = pd.read_parquet('/tmp/users-parquet', engine='pyarrow')
    item_parquet = pd.read_parquet('/tmp/items-parquet', engine='pyarrow')

    user_df = pd.DataFrame(index=range(0, int(user_parquet.customerID.max()) + 1)).join(
        pd.DataFrame(np.stack(user_parquet.flatList), index=user_parquet.customerID).fillna(0))
    item_df = pd.DataFrame(np.stack(item_parquet.itemAffinities).T, columns=item_parquet.itemID).fillna(0)
    df = user_df.dot(item_df).T

    def write_to_parquet(df, filename):
        if not os.path.exists("/tmp/sampleout"):
            os.makedirs("/tmp/sampleout")

        df.columns = ['id', 'ratings']
        df.to_parquet(filename, engine='pyarrow')

    k = int(args[1])
    output_list = []
    part = 0

    for i, r in enumerate(df):
        top_k_df = df[r].sort_values(ascending=False)[:k].index.tolist()
        output_list.append((i, [[index, df[r][index]] for index in top_k_df]))
        if (i + 1) % 25000 == 0:
            write_to_parquet(pd.DataFrame(output_list),
                             "/tmp/sampleout/parquet {}".format(part))
            output_list = []
            part += 1

    if output_list:
        write_to_parquet(pd.DataFrame(output_list),
                         "/tmp/sampleout/parquet {}".format(part))


if __name__ == '__main__':
    main(argv)
