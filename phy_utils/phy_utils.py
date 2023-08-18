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


def intersect(strings: tuple[str, str]) -> str:
    """
    compare two strings in a tuple and returns the
    largest intersection between the two.
    :param strings: tuple with two strings
    :return str:
    """

    sets = ([], [])

    for string_set, string in enumerate(strings):
        for i in range(len(string)):
            for j in range(i, len(string)):
                sets[string_set].append(string[i:j+1])
    
    return max(set(sets[0]) & set(sets[1]), key = len)


def long_format(data: Type[pd.DataFrame], 
                variables: list[str],
                time_points: list[Any]) -> Type[pd.DataFrame]:
    """
    transform a wide tidy dataframe into a long tidy dataframe
    pandas.wide_to_long can be used for this purpose, but it is not
    flexible enough to handle the case where variable names don't end
    with the time point. This function is more flexible and can be used
    for any case where the variable names are not in the format:
    variable_time_point_1, variable_time_point_2, ..., variable_time_point_N

    :param data: wide tidy dataframe
    :param variables: list of strings with the column names
    :param time_points: list of time points
    :return: long tidy dataframe

    """
    
    # extract the suffix that identifies the variable
    column_name = intersect(variables[:2])

    # repeat each observation by len(variables)
    long_data = data.loc[data.index.repeat(len(variables))].reset_index()

    id = 0
    while id  < len(data):

        for entry, variable in enumerate(variables):
            
            long_data.loc[id+entry, column_name] = data.loc[id, variable]
            long_data.loc[id+entry, "time"] = time_points[entry]

        id += 4

    long_data = long_data.drop(columns = variables)

    return long_data

def confidence_interval(measurements: Type[np.ndarray]) -> tuple[float, float]:
    """
    compute 95% confidence intervals for conditinuous and unbounded variables
    assuming a gaussian distribution
    :param measurements: array with measurements
    :return tuple:  
    """
    mean = np.mean(measurements) 
    lower_ci = mean - 1.96 * np.std(measurements)/np.sqrt(len(measurements))
    upper_ci = mean + 1.96 * np.std(measurements)/np.sqrt(len(measurements))

    return lower_ci, upper_ci
