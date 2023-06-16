#include <bits/stdc++.h>

pair<int, int> missingAndRepeating(vector<int> &arr, int n)
{
    vector<int> m(n, 0);
    for (auto it : arr)
    {
        m[it - 1]++;
    }
    int value, value2;
    for (int i = 0; i < n; i++)
    {
        if (m[i] == 0)
        {
            value = i + 1;
        }
        else if (m[i] >= 2)
        {
            value2 = i + 1;
        }
    }
    pair<int, int> ans = {value, value2};
    return ans;
}
