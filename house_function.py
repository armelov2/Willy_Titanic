import pandas as pd
def first_letter(serie):
  return pd.DataFrame(serie.str[0])