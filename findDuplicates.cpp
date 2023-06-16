#include <bits/stdc++.h>

int findDuplicate(vector<int> &arr, int n)
{
    unordered_map<int, int> mp;
    for (auto it : arr)
    {
        mp[it]++;
        if (mp[it] > 1)
            return it;
    }
    return 0;
}
