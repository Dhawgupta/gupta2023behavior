import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

Ps = [ 0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 4.0]
Is = [ 0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 4.0]
Ds = [ 0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 4.0]

def float_to_string(number, precision=10):
    return '{0:.{prec}f}'.format(
        number, prec=precision,
    ).rstrip('0').rstrip('.') or '0'
    
params_best = []
params_good = []
params_bad = []
returns = []
# for p  in Ps:
#     for i in Is:
#         for d in Ds:
            
#             file_name = f'PID_results/{float_to_string(p)}_{float_to_string(i)}_{float_to_string(d)}.npy'
#             arr = np.load(file_name)
#             perf = np.mean(arr)
#             returns.append(perf)
#             param = (p,i,d, perf)
#             if perf > 499:
#                 params_best.append(param)
#             if perf > 250 and perf < 490:
#                 params_good.append(param)
#             if perf < 250:
#                 params_bad.append(param)
#             # print(file_name, perf)
#             # print(arr)

for p  in Ps:
    for d in Ds:
        
        file_name = f'PD_results/{float_to_string(p)}_{float_to_string(d)}.npy'
        arr = np.load(file_name)
        perf = np.mean(arr)
        returns.append(perf)
        param = (p,d, perf)
        if perf > 499:
            params_best.append(param)
        if perf > 250 and perf < 490:
            params_good.append(param)
        if perf < 250:
            params_bad.append(param)


# for p  in Ps:

#     file_name = f'P_results/{float_to_string(p)}.npy'
#     arr = np.load(file_name)
#     perf = np.mean(arr)
#     returns.append(perf)
#     param = (p, perf)
#     if perf > 499:
#         params_best.append(param)
#     if perf > 250 and perf < 490:
#         params_good.append(param)
#     if perf < 250:
#         params_bad.append(param)


params_good.sort(key = lambda x : x[2])
params_bad.sort(key = lambda x : x[2])
print(len(params_best))
print(len(params_good))
print(len(params_bad))
print("BEST")
print(params_best)
print("GOOD")
print(params_good)
print("BAD")
print(params_bad)
# returns = np.array(returns)
# sns.displot(returns)
# plt.plot(returns)
# plt.savefig('pid.png', dpi = 300)

# sns.displot(df, x = 1, kind = "kde", fill = False, hue = 0, palette=agent_colors)
# plt.xlabel('Returns')
# plt.plot(returns)
# plt.savefig('temp.png', dpi = 300)
