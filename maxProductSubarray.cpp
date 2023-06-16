#include <bits/stdc++.h>
int maximumProfit(vector<int> &prices)
{
    int maxi = 0;
    int mini = prices[0];
    for (auto it : prices)
    {
        mini = min(mini, it);
        maxi = max(maxi, it - mini);
    }
    return maxi;
}