from tabulate import tabulate

def orgprint(df):
    print(tabulate(df.reset_index().values, tablefmt="orgtbl",
                    headers=df.reset_index().columns.to_list()))
