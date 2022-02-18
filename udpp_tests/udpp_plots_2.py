import itertools
import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


mincost_color = '#1f77b4'
nnbound_color = '#ff7f0e'
udpp_color = '#2ca02c'


plt.rcParams["figure.figsize"] = (20, 18)
plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.autolayout"] = True

udpp_hfes_ = "udpp_0"
res = pd.read_csv("udpp_tests/cap_n_fl_test_1000_2.csv")
res["reduction"] = res["initial costs"] - res["final costs"]

bins = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100, 204]
clusters = ['1','2','3','4','5-9','10-19','20-29','30-39','40-49', '50-74','75-99','100->']


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



# total costs and reductions
total_inital = df_mincost_total["initial costs"].sum()
mincost_final = df_mincost_total["final costs"].sum()
nnbound_final = df_nnbound_total["final costs"].sum()
udpp_final = df_udpp_total["final costs"].sum()

mincost_total = df_mincost_total.reduction.sum()
nnbound_total = df_nnbound_total.reduction.sum()
udpp_total = df_udpp_total.reduction.sum()

mincost_total_perc = (total_inital - mincost_final ) /total_inital
nnbound_total_perc = (total_inital - nnbound_final ) /total_inital
udpp_total_perc = (total_inital - udpp_final ) /total_inital
udpp_total_perc


plt.rcParams["figure.figsize"] = (10, 18)
plt.bar([1], df_mincost_total.reduction.sum(), width=.8)
plt.bar([2], df_nnbound_total.reduction.sum(), width=.8)
plt.bar([3], df_udpp_total.reduction.sum(), width=.8)
plt.xticks([1, 2, 3], ["MINCOST", "NNBOUND", "UDPP"])
vert_shift = 5000000
plt.annotate('61.5%', (1, mincost_total + vert_shift), ha="center")
plt.annotate('57.0%', (2, nnbound_total + vert_shift), ha="center")
plt.annotate('34.0%', (3, udpp_total + vert_shift), ha="center")
plt.ticklabel_format(style='plain', axis='y')
plt.title("REDUCTION PERCENTAGE")
plt.ylabel("PERCENTAGE")
plt.grid(axis="y")
plt.show()




# percentage comparison
plt.rcParams["figure.figsize"] = (20, 10)

mincost_perc = df_mincost_total["reduction %"].to_numpy()
zeros_idxs = np.where(mincost_perc==0)
mincost_perc[mincost_perc == 0] = 1
udpp_perc = df_udpp_total["reduction %"].to_numpy()
udpp_perc[zeros_idxs[0]] = 1
plt.scatter(df_mincost_total["initial costs"], udpp_perc / mincost_perc)
# plt.plot(range(1000), df_nnbound_total["reduction %"].to_numpy()
#          - df_udpp_total["reduction %"].to_numpy())
plt.ticklabel_format(style='plain')
plt.title("MINCOST - UDPP PERCENTAGE REDUCTION DISTANCE")
plt.xlabel("PERCENTAGE REDUCTION DISTANCE")
plt.ylabel("REDUCTION")
plt.show()




final_value = 1000
plt.rcParams["figure.figsize"] = (20, 10)
fig, ax = plt.subplots()
ax.plot(df_mincost_total["initial costs"].iloc[:final_value], df_mincost_total.reduction.iloc[:final_value], label="MINCOST")
ax.plot(df_mincost_total["initial costs"].iloc[:final_value], df_nnbound_total.reduction.iloc[:final_value],  label="NNBOUND")
ax.plot(df_mincost_total["initial costs"].iloc[:final_value], df_total[df_total.model == udpp_hfes_].reduction.iloc[:final_value],  label="UDPP")
plt.title("INITIAL COST-REDUCTION")
plt.xlabel("INITIAL COST")
plt.ylabel("REDUCTION")
ax.ticklabel_format(style='plain')
plt.grid(axis="y")
plt.legend()
plt.show()






plt.plot(df_mincost_total["initial costs"], df_mincost_total["reduction %"], label="MINCOST")
plt.plot(df_mincost_total["initial costs"], df_nnbound_total["reduction %"], label="NNBOUND")
plt.plot(df_mincost_total["initial costs"], df_total[df_total.model == udpp_hfes_]["reduction %"], label="UDPP")
plt.ticklabel_format(style='plain')
plt.title("INITIAL COST-REDUCTION %")
plt.xlabel("INITIAL COST")
plt.ylabel("REDUCTION %")
plt.grid(axis="y")
plt.legend()
plt.show()


# percentage distance

plt.plot(range(1000), df_mincost_total["reduction %"].to_numpy()
         - df_udpp_total["reduction %"].to_numpy())
plt.plot(range(1000), df_nnbound_total["reduction %"].to_numpy()
         - df_udpp_total["reduction %"].to_numpy())
x_pos = [int(i*999/5) for i in range(6)]
x = [int(df_mincost_total["initial costs"].iloc[i]) for i in x_pos]
plt.xticks(x_pos, x)
plt.show()



# scatter frequency

cap_red = [np.around(i*0.1, decimals=1) for i in range(1,9)]
n_flights = [25*i for i in range(17)]
np.around(0.89, decimals=1)
reg_freq = [df_udpp_total[(df_udpp_total.c_reduction == c) & (df_udpp_total.n_flights >= n_flights[i])
                          & (df_udpp_total.n_flights < n_flights[i+1])].shape[0]
            for c in cap_red for i in range(len(n_flights)-1)]


c_red_n_fl_combs = itertools.product(cap_red, n_flights[1:])

c_red, n_fl = [], []
for i, j in c_red_n_fl_combs:
    c_red.append(i)
    n_fl.append(j)

plt.rcParams["figure.figsize"] = (20, 18)
plt.scatter(c_red, n_fl, s=np.array(reg_freq)*50)
gll = plt.scatter([], [], s=10_000, marker='o', color='#1f77b4')
gl = plt.scatter([], [], s=5_000, marker='o', color='#1f77b4')
ga = plt.scatter([], [], s=1_000, marker='o', color='#1f77b4')
plt.legend((gll, gl, ga), ('10 ML\n\n', '5 ML\n\n', '1 ML'), scatterpoints=1, loc='upper right', ncol=1, fontsize=28)
plt.title("FLIGHTS - CAPACITY REDUCTION FREQUENCY")
plt.xlabel("CAPACITY")
plt.ylabel("N FLIGHTS")
plt.show()




# initial costs


plt.scatter(df_udpp_total.c_reduction, df_udpp_total.n_flights, s=df_udpp_total["initial costs"] * 0.001)
gll = plt.scatter([], [], s=10_000, marker='o', color='#1f77b4')
gl = plt.scatter([], [], s=5_000, marker='o', color='#1f77b4')
ga = plt.scatter([], [], s=1_000, marker='o', color='#1f77b4')
plt.legend((gll, gl, ga), ('10 ML\n\n', '5 ML\n\n', '1 ML'), scatterpoints=1,
           loc='upper right', ncol=1, fontsize=28)
plt.title("FLIGHTS - CAPACITY REDUCTION - INITIAL COSTS")
plt.xlabel("CAPACITY")
plt.ylabel("N FLIGHTS")
plt.show()



plt.scatter(df_udpp_total.c_reduction, df_udpp_total.n_flights, s=df_udpp_total.reduction * 0.001)
gll = plt.scatter([], [], s=10_000, marker='o', color='#1f77b4')
gl = plt.scatter([], [], s=5_000, marker='o', color='#1f77b4')
ga = plt.scatter([], [], s=1_000, marker='o', color='#1f77b4')
plt.legend((gll, gl, ga), ('\n\n10 ML\n', '5 ML\n\n', '1 ML'), scatterpoints=1,
           loc='upper right', ncol=1, fontsize=28)
plt.title("FLIGHTS - CAPACITY REDUCTION - REDUCTION")
plt.xlabel("CAPACITY")
plt.ylabel("N FLIGHTS")
plt.show()


plt.rcParams["figure.figsize"] = (20, 18)
plt.scatter(df_udpp_total.c_reduction, df_udpp_total.n_flights, s=df_udpp_total["initial costs"] * 0.001, edgecolors='black')
plt.scatter(df_udpp_total.c_reduction, df_udpp_total.n_flights, s=df_udpp_total.reduction * 0.001)
gll = plt.scatter([], [], s=10_000, marker='o', color='#1f77b4')
gl = plt.scatter([], [], s=5_000, marker='o', color='#1f77b4')
ga = plt.scatter([], [], s=1_000, marker='o', color='#1f77b4')
plt.legend((gll, gl, ga), ('10 ML\n\n', '5 ML\n\n', '1 ML'), scatterpoints=1,
           loc='upper right', ncol=1, fontsize=28)
plt.title("FLIGHTS - CAPACITY REDUCTION - INITIAL COSTS - reduction")
plt.xlabel("CAPACITY")
plt.ylabel("N FLIGHTS")
plt.show()



plt.scatter(df_udpp_total.c_reduction, df_udpp_total.n_flights, s=(df_udpp_total["reduction %"]) ** 2, c=udpp_color, edgecolors='black')
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

plt.bar([1], df_mincost_total["reduction"].mean(), width=.2, yerr=df_mincost_total["reduction"].std())
plt.bar([2], df_nnbound_total["reduction"].mean(), width=.2, yerr=df_nnbound_total["reduction"].std())
plt.bar([3], df_udpp_total["reduction"].mean(), width=.2, yerr=df_udpp_total["reduction"].std()
)
plt.xticks([1,2,3], ["MINCOST", "NNBOUND", "UDPP"])
plt.ticklabel_format(style='plain', axis='y')
plt.title("REDUCTION")
plt.ylabel("REDUCITON")
plt.grid(axis="y")
plt.show()




df_mincost_total["reduction %"].mean()
df_mincost_total["reduction %"].std()
df_nnbound_total["reduction %"].mean()
df_nnbound_total["reduction %"].std()
df_udpp_total["reduction %"].mean()
df_udpp_total["reduction %"].std()


plt.rcParams["figure.figsize"] = (20, 18)
plt.bar([1], df_mincost_total["reduction %"].mean(), width=.8, yerr=df_mincost_total["reduction %"].std())
plt.bar([2], df_nnbound_total["reduction %"].mean(), width=.8, yerr=df_nnbound_total["reduction %"].std())
plt.bar([3], df_udpp_total["reduction %"].mean(), width=.8, yerr=df_udpp_total["reduction %"].std()
)
plt.xticks([1,2,3], ["MINCOST", "NNBOUND", "UDPP"])
plt.ticklabel_format(style='plain', axis='y')
plt.title("REDUCTION PERCENTAGE")
plt.ylabel("PERCENTAGE")
plt.grid(axis="y")
plt.show()



airlines_dist = df_udpp[df_udpp.airline != "total"]

airlines_dist["num flights"].max()

airlines_counts = airlines_dist["num flights"].value_counts()
df_airlines_counts = pd.DataFrame({"n_flights": airlines_counts.index, "freq": airlines_counts})
df_airlines_counts.sort_values(by="n_flights", inplace=True)
plt.bar(df_airlines_counts.n_flights, df_airlines_counts.freq)
plt.title("N FLIGHTS PER AIRLINE")
plt.xlabel("N FLIGHTS PER AIRLINE")
plt.ylabel("FREQUENCY")
plt.grid(axis="y")
plt.show()





h = plt.hist(airlines_dist["num flights"], bins=bins, rwidth=1)
plt.cla()
plt.bar(range(h[0].shape[0]), h[0])
plt.xticks(range(h[0].shape[0]), clusters)
plt.title("N FLIGHTS PER AIRLINE (CLUSTER)")
plt.xlabel("N FLIGHTS PER AIRLINE CLUSTER")
plt.ylabel("FREQUENCY")
plt.grid(axis="y")
plt.show()
# h[1][:-1]





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



# ********************** reduction

plt.bar(np.array(range(len(mincost_reduction))) - .2, mincost_reduction, width=.2, yerr=mincost_reduction_std)
plt.bar(np.array(range(len(nnbound_reduction))), nnbound_reduction, width=.2, yerr=nnbound_reduction_std)
plt.bar(np.array(range(len(udpp_reduction))) + .2, udpp_reduction, width=.2, yerr=udpp_reduction_std)
plt.xticks(range(len(mincost_reduction)), clusters)
plt.ticklabel_format(style='plain', axis='y')
plt.title("MEAN REDUCTION PER AIRLINE CLUSTER")
plt.xlabel("N FLIGHTS PER AIRLINE (CLUSTER)")
plt.ylabel("REDUCTION")
plt.grid(axis="y")
plt.show()








mincost_reduction_p, mincost_reduction_p_std = get_mean_std(df_mincost_airlines, bins, "reduction %")
nnbound_reduction_p, nnbound_reduction_p_std = get_mean_std(df_nnbound_airlines, bins, "reduction %")
udpp_reduction_p, udpp_reduction_p_std = get_mean_std(df_udpp_airlines, bins, "reduction %")

plt.bar(np.array(range(len(mincost_reduction_p))) - .2, mincost_reduction_p, width=.2, yerr=mincost_reduction_p_std)
plt.bar(np.array(range(len(nnbound_reduction_p))), nnbound_reduction_p, width=.2, yerr=nnbound_reduction_p_std)
plt.bar(np.array(range(len(udpp_reduction_p))) + .2, udpp_reduction_p, width=.2, yerr=udpp_reduction_p_std)
plt.xticks(range(len(mincost_reduction_p)), bins[:-1])
plt.title("MEAN REDUCTION % PER AIRLINE CLUSTER")
plt.xlabel("N FLIGHTS PER AIRLINE (CLUSTER)")
plt.ylabel("REDUCTION %")
plt.grid(axis="y")
plt.show()






plt.bar(np.array(range(len(mincost_reduction_p))) - .2, mincost_reduction_p, width=.2)
plt.bar(np.array(range(len(nnbound_reduction_p))), nnbound_reduction_p, width=.2)
plt.bar(np.array(range(len(udpp_reduction_p))) + .2, udpp_reduction_p, width=.2)
plt.xticks(range(len(mincost_reduction_p)), bins[:-1])
plt.title("MEAN REDUCTION % PER AIRLINE CLUSTER")
plt.xlabel("N FLIGHTS PER AIRLINE (CLUSTER)")
plt.ylabel("REDUCTION %")
plt.grid(axis="y")
plt.show()


# reduction % reduced negative scale

mincost_reduction_p_standard = np.array(mincost_reduction_p)
mincost_reduction_p_standard[mincost_reduction_p_standard < 0] = mincost_reduction_p_standard[mincost_reduction_p_standard < 0] /100
plt.bar(np.array(range(len(mincost_reduction_p_standard))) - .2, mincost_reduction_p_standard, width=.2)
plt.bar(np.array(range(len(nnbound_reduction_p))), nnbound_reduction_p, width=.2)
plt.bar(np.array(range(len(udpp_reduction_p))) + .2, udpp_reduction_p, width=.2)
plt.xticks(range(len(mincost_reduction_p)), clusters)
plt.yticks(range(-10, 61, 10), [-1000, 0, 10, 20, 30, 40, 50, 60])
plt.title("MEAN REDUCTION % PER AIRLINE CLUSTER")
plt.xlabel("N FLIGHTS PER AIRLINE (CLUSTER)")
plt.ylabel("REDUCTION %")
plt.grid(axis="y")
plt.show()




# positive negative impacts

pos = df_udpp_airlines[df_udpp_airlines.positive > 0]

df_udpp_total.protections.sum()
df_udpp_total.positive.sum()
df_udpp_total.negative.sum()
df_udpp_total["positive mins"].sum()
df_udpp_total["negative mins"].sum()

udpp_positive, udpp_positive_std = get_mean_std(df_udpp_airlines, bins, "positive")
plt.bar(np.array(range(len(udpp_positive))) + .2, udpp_positive, width=.2, yerr= udpp_positive_std)
plt.xticks(range(len(udpp_positive)), bins[:-1])
plt.title("MEAN POSITIVE IMPACT INSTANCE PER CLUSTER")
plt.xlabel("CLUSTER")
plt.ylabel("FREQUENCY mean")
plt.grid(axis='y')
plt.show()



udpp_protections, udpp_protections_std = get_mean_std(df_udpp_airlines, bins, "protections")
plt.bar(np.array(range(len(udpp_protections))) + .2, udpp_protections, width=.2)
plt.xticks(range(len(udpp_protections)), bins[:-1])
plt.title("PROTECTION PER CLUSTER")
plt.xlabel("CLUSTER")
plt.ylabel("PROTECTIONS mean")
plt.grid(axis='y')
plt.show()

neg = df_udpp_airlines[df_udpp_airlines.reduction < 0]

negative_mean, negative_std = get_mean_std(neg, bins, "negative")
neg_mean = [val if not math.isnan(val) else 0 for val in negative_mean]
neg_std = [val if not math.isnan(val) else 0 for val in negative_std]
plt.bar(np.array(range(len(neg_mean))) + .2, neg_mean, width=.2, yerr=neg_std)
plt.xticks(range(len(neg_mean)), bins[:-1])
plt.title("NEGATIVE IMPACT PER CLUSTER OCCURRENCES")
plt.xlabel("CLUSTER")
plt.ylabel("FREQUENCY mean")
plt.grid(axis='y')
plt.show()

negative_mean, negative_std = get_mean_std(neg, bins, "reduction")
neg_mean = [val if not math.isnan(val) else 0 for val in negative_mean]
neg_std = [val if not math.isnan(val) else 0 for val in negative_std]
plt.bar(np.array(range(len(neg_mean))) + .2, neg_mean, width=.2, yerr=neg_std)
plt.xticks(range(len(neg_mean)), bins[:-1])
plt.title("NEGATIVE IMPACT PER CLUSTER")
plt.xlabel("CLUSTER")
plt.ylabel("IMPACT mean")
plt.grid(axis='y')
plt.show()

pos.run.unique().shape[0]
neg.run.unique().shape[0]

neg_top = neg[["airline", "num flights", "initial costs", "reduction %", "n_flights", "low_cost", "negative", "negative mins", "airport", "reduction"]].sort_values(by= "reduction")



# ******************************* hfes 5

res_5 = pd.read_csv("udpp_tests/cap_n_fl_test_1000_hfes.csv")
res_5["reduction"] = res_5["initial costs"] - res_5["final costs"]

hfes_5 = "udpp_5"
df_udpp = res[res.model == hfes_5]

df_total_5 = res_5[res_5.airline == "total"]
df_airlines_5 = res_5[res_5.airline != "total"]
df_total_5 = df_total_5.sort_values(by="initial costs")
df_udpp_total_5 = df_total_5[df_total_5.model == hfes_5]




plt.rcParams["figure.figsize"] = (20, 10)
fig, ax = plt.subplots()
ax.plot(df_mincost_total["initial costs"], df_total[df_total.model == udpp_hfes_].reduction,  label="UDPP 0")
ax.plot(df_mincost_total["initial costs"], df_total_5[df_total_5.model == hfes_5].reduction,  label="UDPP 5")
plt.title("UDPP_0 vs UDPP_5")
plt.xlabel("INITIAL COST")
plt.ylabel("REDUCTION")
ax.ticklabel_format(style='plain')
plt.grid(axis="y")
plt.legend()
plt.show()



df_udpp_airlines_5 = df_airlines_5[df_airlines_5.model == hfes_5]



neg_5 = df_udpp_airlines_5[df_udpp_airlines_5.reduction < 0]

negative_mean, negative_std = get_mean_std(neg_5, bins, "negative")
neg_mean = [val if not math.isnan(val) else 0 for val in negative_mean]
neg_std = [val if not math.isnan(val) else 0 for val in negative_std]
plt.bar(np.array(range(len(neg_mean))) + .2, neg_mean, width=.2, yerr=neg_std)
plt.xticks(range(len(neg_mean)), bins[:-1])
plt.title("NEGATIVE IMPACT PER CLUSTER OCCURRENCES (HFES=5)")
plt.xlabel("CLUSTER")
plt.ylabel("FREQUENCY mean")
plt.grid(axis='y')
plt.show()

negative_mean, negative_std = get_mean_std(neg_5, bins, "reduction")
neg_mean = [val if not math.isnan(val) else 0 for val in negative_mean]
neg_std = [val if not math.isnan(val) else 0 for val in negative_std]
plt.bar(np.array(range(len(neg_mean))) + .2, neg_mean, width=.2, yerr=neg_std)
plt.xticks(range(len(neg_mean)), bins[:-1])
plt.title("NEGATIVE IMPACT PER CLUSTER (HFES=5)")
plt.xlabel("CLUSTER")
plt.ylabel("IMPACT mean")
plt.grid(axis='y')
plt.show()

pos.run.unique().shape[0]
neg.run.unique().shape[0]

neg_top_5 = neg_5[["airline", "num flights", "initial costs", "reduction %", "n_flights", "low_cost", "negative", "negative mins", "airport", "reduction"]].sort_values(by= "reduction")


df_udpp_airlines["comp time"].mean()
df_udpp_airlines["comp time"].std()
df_udpp_airlines["comp time"].max()



comp_time = df_udpp_airlines[["num flights", "comp time"]]



# curfew and missed conntecting

udpp_hfes_ = "udpp_0"
res = pd.read_csv("udpp_tests/cap_n_fl_test_1000_hfes_curfew.csv")
res["reduction"] = res["initial costs"] - res["final costs"]
df_udpp = res[res.model == "udpp_5"]

df_udpp_total_curfew = df_udpp[df_udpp.airline == "total"]

init_mis_con = df_udpp_total_curfew.init_mis_con.sum()
final_mis_con = df_udpp_total_curfew.final_mis_con.sum()

1 - final_mis_con/init_mis_con


init_curfew = df_udpp_total_curfew.init_curfew.sum()
final_curfew = df_udpp_total_curfew.final_curfew.sum()

1 - final_curfew/init_curfew

plt.rcParams["figure.figsize"] = (10, 18)
plt.bar([1], init_mis_con, width=.2, color='grey')
plt.bar([1.3], final_mis_con, width=.2, color=udpp_color)
plt.xticks([1,1.3], ["INITIAL", "FINAL"])
plt.ticklabel_format(style='plain', axis='y')
plt.title("MISSED CONNECTING")
plt.ylabel("NUM PAX")
plt.grid(axis="y")
plt.show()


plt.rcParams["figure.figsize"] = (10, 18)
plt.bar([1], init_curfew, width=.2, color='grey')
plt.bar([1.3], final_curfew, width=.2, color=udpp_color)
plt.xticks([1,1.3], ["INITIAL", "FINAL"])
plt.ticklabel_format(style='plain', axis='y')
plt.title("HITTING CURFEW")
plt.ylabel("NUM FLIGHTS")
plt.grid(axis="y")
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


pax = pd.read_csv("ScenarioAnalysis/Pax/pax.csv")
pp = pax[['pax', 'delta_leg1', 'delay', 'airline', 'origin', 'destination', 'air_cluster']].iloc[4:8]