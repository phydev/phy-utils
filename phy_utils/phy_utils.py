import seaborn.objects as so
from typing import Type

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
