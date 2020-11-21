from mlbc.general.utils import *

# Read the data and clean it up.
df = read_data('data/cars.csv')
clean_alphanum_data(df)


