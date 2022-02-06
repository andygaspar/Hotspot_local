import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (20, 18)
plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.autolayout"] = True

udpp_hfes_ = "udpp_0"
res = pd.read_csv("udpp_tests/cap_n_fl_test_1000_2.csv")
res["reduction"] = res["initial costs"] - res["final costs"]

# p = res[(res.run == 345) & (res.model == "udpp_0")]


df_mincost = res[res.model == "mincost"]
df_nnbound = res[res.model == "nnbound"]
df_udpp = res[res.model == udpp_hfes_]

df_total = res[res.airline == "total"]
df_airlines = res[res.airline != "total"]
df_total = df_total.sort_values(by="initial costs")

df_mincost_total = df_total[df_total.model == "mincost"]
df_nnbound_total = df_total[df_total.model == "nnbound"]
df_udpp_total = df_total[df_total.model == udpp_hfes_]

fig, ax = plt.subplots()
ax.plot(df_mincost_total["initial costs"], df_mincost_total.reduction)
ax.plot(df_mincost_total["initial costs"], df_nnbound_total.reduction)
ax.plot(df_mincost_total["initial costs"], df_total[df_total.model == udpp_hfes_].reduction)
plt.title("INITIAL COST-REDUCTION")
plt.xlabel("INITIAL COST")
plt.ylabel("REDUCTION")
ax.ticklabel_format(style='plain')
plt.show()

plt.plot(df_mincost_total["initial costs"], df_mincost_total["reduction %"])
plt.plot(df_mincost_total["initial costs"], df_nnbound_total["reduction %"])
plt.plot(df_mincost_total["initial costs"], df_total[df_total.model == udpp_hfes_]["reduction %"])
plt.ticklabel_format(style='plain')
plt.title("INITIAL COST-REDUCTION %")
plt.xlabel("INITIAL COST")
plt.ylabel("REDUCTION %")
plt.show()

plt.scatter(df_udpp_total.c_reduction, df_udpp_total.n_flights, s=df_udpp_total.reduction * 0.001)
gll = plt.scatter([], [], s=10_000, marker='o', color='#1f77b4')
gl = plt.scatter([], [], s=5_000, marker='o', color='#1f77b4')
ga = plt.scatter([], [], s=1_000, marker='o', color='#1f77b4')
plt.legend((gll, gl, ga), ('10 ML\n\n', '5 ML\n\n', '1 ML'), scatterpoints=1,
           loc='upper right', ncol=1, fontsize=28)
plt.title("FLIGHTS - CAPACITY REDUCTION - REDUCTION")
plt.xlabel("CAPACITY")
plt.ylabel("N FLIGHTS")
plt.show()






plt.scatter(df_udpp_total.c_reduction, df_udpp_total.n_flights, s=(df_udpp_total["reduction %"]) ** 2)
gll = plt.scatter([], [], s=10_000, marker='o', color='#1f77b4')
gl = plt.scatter([], [], s=2_500, marker='o', color='#1f77b4')
ga = plt.scatter([], [], s=1_000, marker='o', color='#1f77b4')
plt.legend((gll, gl, ga), ('100%\n\n', '50%\n\n', '10%'), scatterpoints=1,
           loc='upper right', ncol=1, fontsize=28)
plt.title("FLIGHTS - CAPACITY REDUCTION - REDUCTION %")
plt.xlabel("CAPACITY")
plt.ylabel("N FLIGHTS")
plt.show()





df_mincost_total["reduction %"].mean()
df_mincost_total["reduction %"].std()
df_nnbound_total["reduction %"].mean()
df_nnbound_total["reduction %"].std()
df_udpp_total["reduction %"].mean()
df_udpp_total["reduction %"].std()

airlines_dist = df_udpp[df_udpp.airline != "total"]

airlines_dist["num flights"].max()

airlines_counts = airlines_dist["num flights"].value_counts()
df_airlines_counts = pd.DataFrame({"n_flights": airlines_counts.index, "freq": airlines_counts})
df_airlines_counts.sort_values(by="n_flights", inplace=True)
plt.plot(df_airlines_counts.n_flights, df_airlines_counts.freq)
plt.title("N FLIGHTS PER AIRLINE")
plt.xlabel("N FLIGHTS PER AIRLINE")
plt.ylabel("FREQUENCY")
plt.show()





bins = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100, 204]
h = plt.hist(airlines_dist["num flights"], bins=bins, rwidth=1)
plt.cla()
plt.bar(range(h[0].shape[0]), h[0])
plt.xticks(range(h[0].shape[0]), h[1][:-1])
plt.title("N FLIGHTS PER AIRLINE (CLUSTER)")
plt.xlabel("N FLIGHTS PER AIRLINE CLUSTER")
plt.ylabel("FREQUENCY")
plt.show()






df_mincost_airlines = df_airlines[df_airlines.model == "mincost"]
df_nnbound_airlines = df_airlines[df_airlines.model == "nnbound"]
df_udpp_airlines = df_airlines[df_airlines.model == udpp_hfes_]


def get_mean_std(df, bins_, item):
    mean_reduction = [df[(bins_[i] <= df["num flights"]) & (df["num flights"] < bins_[i + 1])][item].mean()
                      for i in range(len(bins_) - 1)]
    std_reduction = [df[(bins_[i] <= df["num flights"]) & (df["num flights"] < bins_[i + 1])][item].std()
                     for i in range(len(bins_) - 1)]
    return mean_reduction, std_reduction


mincost_reduction, mincost_reduction_std = get_mean_std(df_mincost_airlines, bins, "reduction")
nnbound_reduction, nnbound_reduction_std = get_mean_std(df_nnbound_airlines, bins, "reduction")
udpp_reduction, udpp_reduction_std = get_mean_std(df_udpp_airlines, bins, "reduction")


plt.bar(np.array(range(len(mincost_reduction))) - .2, mincost_reduction, width=.2, yerr=mincost_reduction_std)
plt.bar(np.array(range(len(nnbound_reduction))), nnbound_reduction, width=.2, yerr=nnbound_reduction_std)
plt.bar(np.array(range(len(udpp_reduction))) + .2, udpp_reduction, width=.2, yerr=udpp_reduction_std)
plt.xticks(range(len(mincost_reduction)), bins[:-1])
plt.ticklabel_format(style='plain', axis='y')
plt.title("REDUCTION PER AIRLINE CLUSTER")
plt.xlabel("N FLIGHTS PER AIRLINE (CLUSTER)")
plt.ylabel("REDUCTION")
plt.show()




mincost_reduction_p, mincost_reduction_p_std = get_mean_std(df_mincost_airlines, bins, "reduction %")
nnbound_reduction_p, nnbound_reduction_p_std = get_mean_std(df_nnbound_airlines, bins, "reduction %")
udpp_reduction_p, udpp_reduction_p_std = get_mean_std(df_udpp_airlines, bins, "reduction %")

plt.bar(np.array(range(len(mincost_reduction_p))) - .2, mincost_reduction_p, width=.2, yerr=mincost_reduction_p_std)
plt.bar(np.array(range(len(nnbound_reduction_p))), nnbound_reduction_p, width=.2, yerr=nnbound_reduction_p_std)
plt.bar(np.array(range(len(udpp_reduction_p))) + .2, udpp_reduction_p, width=.2, yerr=udpp_reduction_p_std)
plt.xticks(range(len(mincost_reduction_p)), bins[:-1])
plt.title("REDUCTION % PER AIRLINE CLUSTER")
plt.xlabel("N FLIGHTS PER AIRLINE (CLUSTER)")
plt.ylabel("REDUCTION %")
plt.show()

plt.bar(np.array(range(len(mincost_reduction_p))) - .2, mincost_reduction_p, width=.2)
plt.bar(np.array(range(len(nnbound_reduction_p))), nnbound_reduction_p, width=.2)
plt.bar(np.array(range(len(udpp_reduction_p))) + .2, udpp_reduction_p, width=.2)
plt.xticks(range(len(mincost_reduction_p)), bins[:-1])
plt.title("REDUCTION % PER AIRLINE CLUSTER")
plt.xlabel("N FLIGHTS PER AIRLINE (CLUSTER)")
plt.ylabel("REDUCTION %")
plt.show()




# positive negative impacts

df_udpp_total.protections.sum()
df_udpp_total.positive.sum()
df_udpp_total.negative.sum()
df_udpp_total["positive mins"].sum()
df_udpp_total["negative mins"].sum()

udpp_positive, udpp_positive_std = get_mean_std(df_udpp_airlines, bins, "positive")
plt.bar(np.array(range(len(udpp_positive))) + .2, udpp_positive, width=.2)
plt.xticks(range(len(udpp_positive)), bins[:-1])
plt.title("POSITIVE IMPACT PER CLUSTER")
plt.xlabel("CLUSTER")
plt.ylabel("FREQUENCY mean")
plt.show()



udpp_protections, udpp_protections_std = get_mean_std(df_udpp_airlines, bins, "protections")
plt.bar(np.array(range(len(udpp_protections))) + .2, udpp_protections, width=.2)
plt.xticks(range(len(udpp_protections)), bins[:-1])
plt.title("PROTECTION PER CLUSTER")
plt.xlabel("CLUSTER")
plt.ylabel("PROTECTIONS mean")
plt.show()

neg = df_udpp_airlines[df_udpp_airlines.reduction < 0]

negative_mean, negative_std = get_mean_std(neg, bins, "negative")
neg_mean = [val if not math.isnan(val) else 0 for val in negative_mean]
neg_std = [val if not math.isnan(val) else 0 for val in negative_std]
plt.bar(np.array(range(len(neg_mean))) + .2, neg_mean, width=.2, yerr=neg_std)
plt.xticks(range(len(neg_mean)), bins[:-1])
plt.title("NEGATIVE IMPACT PER CLUSTER")
plt.xlabel("CLUSTER")
plt.ylabel("FREQUENCY mean")
plt.show()

negative_mean, negative_std = get_mean_std(neg, bins, "reduction")
neg_mean = [val if not math.isnan(val) else 0 for val in negative_mean]
neg_std = [val if not math.isnan(val) else 0 for val in negative_std]
plt.bar(np.array(range(len(neg_mean))) + .2, neg_mean, width=.2, yerr=neg_std)
plt.xticks(range(len(neg_mean)), bins[:-1])
plt.title("NEGATIVE IMPACT PER CLUSTER")
plt.xlabel("CLUSTER")
plt.ylabel("IMPACT mean")
plt.show()

# # num flights **********************************
#
#
# num_flights_means = []
# num_flights_std = []
#
# for airline in airlines:
#     df_air = res[res.airline == airline]
#     num_flights_means.append(df_air["num flights"].mean())
#     num_flights_std.append(df_air["num flights"].std())
#
# plt.rcParams["figure.figsize"] = (20, 18)
# plt.rcParams.update({'font.size': 22})
#
# x_pos = range(6)
# fig, ax = plt.subplots()
# # ax.yaxis.grid(True, zorder=0)
# ax.bar(x_pos, num_flights_means, yerr=num_flights_std, align='center', alpha=1, ecolor='black', capsize=10, zorder=3)
#
# ax.set_xticks(x_pos)
# ax.set_xticklabels(airlines)
# plt.tight_layout()
# plt.show()
#
# # reductions ****************************************************
#
# df_tot = res[res.airline == "total"]
# tot_means = []
# tot_stds = []
# labels = []
# for model in res.model.unique():
#     df_mod = df_tot[df_tot.model == model]
#     tot_means.append(df_mod["reduction %"].mean())
#     tot_stds.append(df_mod["reduction %"].std())
#     labels.append(model)
#
# x_pos = range(len(labels))
# fig, ax = plt.subplots()
#
# ax.bar(x_pos, tot_means, yerr=tot_stds, align='center', alpha=1, ecolor='black', capsize=10, zorder=3)
#
# ax.set_xticks(x_pos)
# ax.set_xticklabels(labels)
#
# # Save the figure and show
# plt.tight_layout()
# # plt.savefig('bar_plot_with_error_bars.png')
# plt.show()
#
# res[(res.model == "udpp") & (res.airline == "total")]["reduction %"].std()
#
# # per airlines
#
# udpp = []
# udpp_std = []
# for airline in airlines:
#     df_air = res[res.airline == airline]
#     df_air_udpp = df_air[df_air.model == "udpp"]
#     udpp.append(df_air_udpp["reduction %"].mean())
#     udpp_std.append(df_air_udpp["reduction %"].std())
#     # istop.append(df_air_istop["reduction %"].mean())
#
# x_pos = range(len(airlines))
# fig, ax = plt.subplots()
#
# # ax.bar(x_pos, istop, align='center', alpha=1, ecolor='black', capsize=10)
# ax.bar(x_pos, udpp, yerr=udpp_std, align='center', alpha=1, ecolor='black', capsize=10)
# ax.set_xticks(x_pos)
# ax.set_xticklabels(airlines)
#
# # Save the figure and show
# plt.tight_layout()
# # plt.savefig('bar_plot_with_error_bars.png')
# plt.show()
#
# udpp
