import numpy as np

def act(x):
    return 0 if x < 0.5 else 1

def go(House, Quarter, Underground):
    x = np.array([House, Quarter, Underground])
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]
    weight1 = np.array([w11, w12])
    weight2 = np.array([-1, 1])

    sum_hidden = np.dot(weight1, x)
    print("Значения сумм на нейронах скрытого слоя: " + str(sum_hidden))

    out_hidden = np.array([act(x) for x in sum_hidden])
    print("Значения на выходах нейронов скрытого слоя: " + str(out_hidden))

    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)
    print ("Выходное значение НС: " + str(y))

    return y


House = 1
Quarter = 1
attr = 0

res = go(House, Quarter, attr)
if res == 1:
    print("Куплю")
else:
    print ("Не куплю")