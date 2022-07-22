import pandas as pd


# def datagram_to_zarr(list_dgrams, vars_n_coords):
#
#
#     vars_n_coords = {"power": ["timestamp", "frequency"],
#                      "angle": ["timestamp", "frequency"]}
#
#     datagram_df = pd.DataFrame.from_dict(list_dgrams)
#
#     # create multi index using the product of the unique
#     # timestamps and channels
#     time_unique = list(data_gram_df["timestamp"].unique())
#     freq_unique = list(np.sort(data_gram_df["frequency"].unique()))
#     # set index as the timestamps and channels
#     data_gram_df_interm = data_gram_df.set_index(["timestamp", "frequency"])
#
#     new_index = pd.MultiIndex.from_product([time_unique, freq_unique], names=["ping_time", "frequency"])
#
#     # Pad dataframe with NaNs using the new multi index
#     data_gram_df = data_gram_df_interm.reindex(new_index, fill_value=np.nan)




