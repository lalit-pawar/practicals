def printJobScheduling(arr, t):
    arr.sort(key=lambda x: x[2], reverse=True)
    result = [False] * t
    job = ['-1'] * t

    for i in arr:
        for j in range(min(t - 1, i[1] - 1), -1, -1):
            if not result[j]:
                result[j] = True
                job[j] = i[0]
                break

    print("Following is the maximum profit sequence of jobs:")
    print(job)

if __name__ == '__main__':
    arr = [['a', 2, 100], ['b', 1, 19], ['c', 2, 27], ['d', 1, 25], ['e', 3, 15]]
    printJobScheduling(arr, 3)
