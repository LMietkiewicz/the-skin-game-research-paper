#%%
import pandas as pd

df_new = pd.read_excel(r"C:\Users\lukas\Desktop\skin_disease_literature_review_iteration_3.xlsx")
df_old = pd.read_excel(r"C:\Users\lukas\Desktop\skin_disease_literature_review_NEW_CURATED_FINAL.xlsx")

#%%
df_new.info()
# %%
df_old.info()
# %%
df_old.drop(columns=["Unnamed: 20", "Problems"], inplace=True)
# %%
df_new.drop(columns=["Study Type"], inplace=True)
# %%
df_old["Paper_Index"].value_counts().index
# %%
df_new["Paper_Index"].value_counts().index
# %%
set_old = set(df_old["Original_Study_Name"].value_counts().index) 
set_new = set(df_new["Original_Study_Name"].value_counts().index)

inner = set_old.intersection(set_new)
inner
# %%
df_old_curated = df_old[df_old["Original_Study_Name"].isin(inner)]
df_new_curated = df_new[df_new["Original_Study_Name"].isin(inner)]
# %%
df_old_curated.info()
# %%
df_new_curated.info()
# %%
df_new_curated.to_excel(r"C:\Users\lukas\Desktop\skin_disease_literature_review_ITERATION_3_CURATED_FINAL.xlsx", index=False)
# %%
df_new_curated.fillna("N/A", inplace=True)
# %%
df_new_curated = df_new_curated[df_new_curated["Paper Type"] != "literature_review"]
# %%
