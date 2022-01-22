import pandas as pd

df_curfew = pd.read_csv("Curfew/Curfew_201902.csv")
df_curfew_eurocontrol = pd.read_csv("Curfew/curefew_eurocontrol.csv")

df_curfew_eurocontrol.OpenHour = df_curfew_eurocontrol.OpenHour.apply(lambda t: int(t[:2])*60 + int(t[3:]))
df_curfew_eurocontrol.CloseHour = df_curfew_eurocontrol.CloseHour.apply(lambda t: int(t[:2])*60 + int(t[3:]))


df_curfew = df_curfew[df_curfew.Year == 2019]
df_curfew = df_curfew[df_curfew.CloseHour != 24]
df_curfew.OpenHour = df_curfew.OpenHour.apply(lambda t: t*60)
df_curfew.CloseHour = df_curfew.CloseHour.apply(lambda t: t*60)


df_curfew = df_curfew[~df_curfew.Airport.isin(df_curfew_eurocontrol.Airport)]
df_curfew = pd.concat([df_curfew, df_curfew_eurocontrol], ignore_index=True)
df_curfew = df_curfew[["Airport", "OpenHour", "CloseHour"]]
df_curfew.to_csv("Curfew/curfew.csv", index_label=False, index=False)
