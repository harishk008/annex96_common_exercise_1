import numpy as np

def nmbe(y, y_ref):
    return float(np.mean(y - y_ref) / np.mean(y_ref) * 100)

def cvrmse(y, y_ref):
    return float(np.sqrt(np.mean((y - y_ref) ** 2)) / np.mean(y_ref) * 100)

def comfort_violation(temps, lo=22.0, hi=26.0):
    return float(np.mean((temps < lo) | (temps > hi)) * 100)

def compute_kpis(district_target, net_load, building_temps):
    T      = len(net_load)
    target = district_target[:T]
    valid  = target > 0
    return {
        "NMBE [%]"             : round(nmbe(net_load[valid], target[valid]), 3),
        "CV-RMSE [%]"          : round(cvrmse(net_load[valid], target[valid]), 3),
        "Temp Comfort violation [%]": round(comfort_violation(building_temps), 2),
    }