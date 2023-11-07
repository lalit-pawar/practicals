def isSubsetSumDP(arr, target):
    dp = [set() for _ in range(target + 1)]
    dp[0].add(tuple())

    for num in arr:
        for j in range(target, num - 1, -1):
            for subset in dp[j - num]:
                dp[j].add(subset + (num,))

    return dp[target]

def printAllPairs(subsets):
    for pair in subsets:
        print(pair)

# Test the code
arr = [3, 7, 4, 6, 5, 2]
target = 9

print("Dynamic Programming Approach:")
result = isSubsetSumDP(arr, target)

if result:
    print("Pairs of subsets that add up to the target:")
    printAllPairs(result)
else:
    print("No subset found with the given target sum.")

