import pandas as pd

def append_to_df(model, name, print_performance=False):

    if name != "udpp_0":
        model.report["model"] = name
        model.report["comp time"] = None
        model.report["protections"] = None
        model.report["positive"] = None
        model.report["positive mins"] = None
        model.report["negative"] = None
        model.report["negative mins"] = None

    else:
        model.report["model"] = "udpp_0"
        model.report["comp time"] = [model.computationalTime] + \
                                            [airline.udppComputationalTime for airline in model.airlines]

        protections = [airline.protections for airline in model.airlines]
        model.report["protections"] = [sum(protections)] + protections

        positive = [airline.positiveImpact for airline in model.airlines]
        positiveMins = [airline.positiveImpactMinutes for airline in model.airlines]

        model.report["positive"] = [sum(positive)] + positive
        model.report["positive mins"] = [sum(positiveMins)] + positiveMins

        negative = [airline.negativeImpact for airline in model.airlines]
        negativeMins = [airline.negativeImpactMinutes for airline in model.airlines]

        model.report["negative"] = [sum(negative)] + negative
        model.report["negative mins"] = [sum(negativeMins)] + negativeMins

    if print_performance:
        model.print_performance()


def append_results(df, global_model, max_model, udpp_model_xp, i, n_flights, c_reduction, airport, print_df=False):
    df_run = global_model.report
    df_run = pd.concat([df_run, max_model.report], ignore_index=True)
    df_run = pd.concat([df_run, udpp_model_xp.report], ignore_index=True)
    df_run["run"] = i
    df_run["n_flights"] = n_flights
    df_run["c_reduction"] = c_reduction
    df_run["airport"] = airport
    if print_df:
        print(df_run, "\n")

    return pd.concat([df, df_run], ignore_index=True)
