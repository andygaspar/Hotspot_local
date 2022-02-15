import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.autolayout"] = True


res = pd.read_csv("udpp_tests/cap_n_fl_test_1000.csv")
res["reduction"] = res["initial costs"] - res["final costs"]

df_mincost = res[res.model == "mincost"]
df_nnbound = res[res.model == "nnbound"]
df_udpp = res[res.model == "udpp_0"]

df_total = res[res.airline == "total"]
df_total = df_total.sort_values(by="initial costs")

df_mincost_total = df_total[df_total.model == "mincost"]
df_nnbound_total = df_total[df_total.model == "nnbound"]
df_udpp_total = df_total[df_total.model == "udpp_0"]



plt.plot(df_mincost_total["initial costs"], df_mincost_total.reduction)
plt.plot(df_mincost_total["initial costs"], df_nnbound_total.reduction)
plt.plot(df_mincost_total["initial costs"], df_total[df_total.model == "udpp_0"].reduction)
plt.show()


plt.plot(df_mincost_total["initial costs"], df_mincost_total["reduction %"])
plt.plot(df_mincost_total["initial costs"], df_nnbound_total["reduction %"])
plt.plot(df_mincost_total["initial costs"], df_total[df_total.model == "udpp_0"]["reduction %"])
plt.show()


plt.plot(range(1000), df_mincost_total["reduction %"].to_numpy()
         - df_udpp_total["reduction %"].to_numpy())
plt.plot(range(1000), df_nnbound_total["reduction %"].to_numpy()
         - df_udpp_total["reduction %"].to_numpy())
x_pos = [int(i*999/5) for i in range(6)]
x = [int(df_mincost_total["initial costs"].iloc[i]) for i in x_pos]
plt.xticks(x_pos, x)
plt.show()



plt.rcParams["figure.figsize"] = (20,18)


plt.scatter(df_udpp_total.c_reduction, df_udpp_total.n_flights, s=df_udpp_total.reduction*0.001)
plt.show()


plt.scatter(df_udpp_total.c_reduction, df_udpp_total.n_flights, s=df_udpp_total["reduction %"]*4)
plt.show()

df_mincost_total["reduction %"].mean()
df_mincost_total["reduction %"].std()
df_nnbound_total["reduction %"].mean()
df_nnbound_total["reduction %"].std()
df_udpp_total["reduction %"].mean()
df_udpp_total["reduction %"].std()
















# num flights **********************************


num_flights_means = []
num_flights_std = []

for airline in airlines:
    df_air = res[res.airline == airline]
    num_flights_means.append(df_air["num flights"].mean())
    num_flights_std.append(df_air["num flights"].std())

plt.rcParams["figure.figsize"] = (20,18)
plt.rcParams.update({'font.size': 22})



x_pos = range(6)
fig, ax = plt.subplots()
# ax.yaxis.grid(True, zorder=0)
ax.bar(x_pos, num_flights_means, yerr=num_flights_std, align='center', alpha=1, ecolor='black', capsize=10, zorder = 3)

ax.set_xticks(x_pos)
ax.set_xticklabels(airlines)
plt.tight_layout()
plt.show()




# reductions ****************************************************

df_tot = res[res.airline == "total"]
tot_means = []
tot_stds = []
labels = []
for model in res.model.unique():
    df_mod = df_tot[df_tot.model == model]
    tot_means.append(df_mod["reduction %"].mean())
    tot_stds.append(df_mod["reduction %"].std())
    labels.append(model)

x_pos = range(len(labels))
fig, ax = plt.subplots()

ax.bar(x_pos, tot_means, yerr=tot_stds, align='center', alpha=1, ecolor='black', capsize=10, zorder=3)

ax.set_xticks(x_pos)
ax.set_xticklabels(labels)


# Save the figure and show
plt.tight_layout()
# plt.savefig('bar_plot_with_error_bars.png')
plt.show()


res[(res.model == "udpp") & (res.airline == "total")]["reduction %"].std()


# per airlines

udpp = []
udpp_std = []
for airline in airlines:
    df_air = res[res.airline == airline]
    df_air_udpp = df_air[df_air.model == "udpp"]
    udpp.append(df_air_udpp["reduction %"].mean())
    udpp_std.append(df_air_udpp["reduction %"].std())
    # istop.append(df_air_istop["reduction %"].mean())



x_pos = range(len(airlines))
fig, ax = plt.subplots()


# ax.bar(x_pos, istop, align='center', alpha=1, ecolor='black', capsize=10)
ax.bar(x_pos, udpp, yerr=udpp_std, align='center', alpha=1, ecolor='black', capsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(airlines)


# Save the figure and show
plt.tight_layout()
# plt.savefig('bar_plot_with_error_bars.png')
plt.show()

udpp
