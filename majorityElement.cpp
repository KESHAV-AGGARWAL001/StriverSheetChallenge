#include <bits/stdc++.h>

int findMajorityElement(int nums[], int n)
{
    unordered_map<int, int> mp;
    int value = 1e8;
    for (int i = 0; i < n; i++)
    {
        mp[nums[i]]++;
        if (mp[nums[i]] > n / 2)
        {
            value = nums[i];
        }
    }
    return value == 1e8 ? -1 : value;
}