n = int(input())
res = []
result = 0
for i in range(n):
    res.append(int(input()))

n_index = res.index(n)
sorted_res = sorted(res)
bias = len(res) - n_index - 1

for i in range(len(res)):
    if sorted_res[i] != res[i - bias]:
        result += 1

print(result - 1)