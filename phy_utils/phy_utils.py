import numpy as np
import pandas as pd
import seaborn.objects as so
from typing import Type, Any

def repeated_measurement_plot(*args,
                              data: Type[pd.DataFrame], 
                              y: list,
                              x: list, 
                              observation: int 
                              ) -> Type[so.Plot]:
    
    """
    Wrapping function for seaborn.objects.Plot to visualize 
    repeated measurement data in tidy wide format 
    Example:
    variable_time_point_1, variable_time_point_2, ..., variable_time_point_N
    30, 40, ..., 50
    70, 50, ..., 50
    ...
    
    :param *args: any extra argument accepted by seaborn.object.Plot
    :param data pd.DataFrame: wide pandas dataframe
    :param y list: list of strings with the column names
    :param x list: list of time points
    :param observation int: which observation (row) should be visualized 
    :return seaborn.object.Plot: seaborn object oriented API
    """

    transformed_data = {
                        "y": np.zeros(len(y)),  
                        "x": x
                        }

    for entry, variable in enumerate(y):
        transformed_data["y"][entry] = data.loc[observation, variable]

    transformed_data = pd.DataFrame(data = transformed_data)

    return so.Plot(*args, data = transformed_data, y = "y", x = "x")


def sample_and_shuffle(file: str, 
                       frac: float = 0.1, 
                       save_to_file: bool = False, 
                       **kwargs: Any) -> Type[pd.DataFrame]:
    """
    shuffle each column in a given dataframe and draw n samples
    this is useful for developing methods out of the server for sensitive data
    and testing on a data file that keeps the same structure
    :param file: path to csv file
    :param frac: fraction of data to sample
    :param save_to_file: save the sampled data to a csv file
    :param kwargs: keyword arguments for pandas.read_csv
    :return: sampled dataframe
    """
    
    # read data
    df = pd.read_csv(file, **kwargs)

    # sample data
    df_shuffled = df.sample(frac=frac)

    # shuffle each column
    for n, column in enumerate(df.columns):

        df_shuffled[column] = df[column].sample(frac=frac).values
    
    # check for data breach on all sampled records
    breach = False
    for record_id in df_shuffled["record_id"].values:
        print(record_id)
        if (df[df["record_id"]==record_id].values == df_shuffled[df_shuffled["record_id"]==record_id].values).all():
            print("data breach on record:", record_id)
            breach = True
            break 
           
    # save to file
    if not breach and save_to_file:
        df_shuffled.to_csv("fake_data.csv", sep=";", index=False)
        print("Success!")
    
    return df_shuffled


def train_test_split_tensors(X, y, **options):
    """
    encapsulation for the sklearn.model_selection.train_test_split function
    in order to split tensors objects and return tensors as output
    :param X: tensorflow.Tensor object
    :param y: tensorflow.Tensor object
    :dict **options: typical sklearn options are available, such as test_size and train_size
    """

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), **options)

    X_train, X_test = tf.constant(X_train), tf.constant(X_test)
    y_train, y_test = tf.constant(y_train), tf.constant(y_test)

    del(train_test_split)

    return X_train, X_test, y_train, y_test
