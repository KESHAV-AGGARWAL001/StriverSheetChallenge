#include <bits/stdc++.h>
long long maxSubarraySum(int nums[], int n)
{
    long long sum = 0;
    long long maxi = -1e9;
    for (int i = 0; i < n; i++)
    {
        sum += nums[i];
        maxi = max(maxi, sum);
        if (sum < 0)
            sum = 0;
    }
    return (maxi < 0) ? 0 : maxi;
}