#include <bits/stdc++.h>
/*

    intervals[i][0] = start point of i'th interval
    intervals[i][1] = finish point of i'th interval

*/

vector<vector<int>> mergeIntervals(vector<vector<int>> &intervals)
{
    sort(intervals.begin(), intervals.end());
    vector<vector<int>> result;
    vector<int> pr = intervals[0];
    for (auto it : intervals)
    {
        if (pr[1] >= it[0])
        {
            pr[1] = max(pr[1], it[1]);
        }
        else
        {
            result.emplace_back(pr);
            pr = it;
        }
    }
    result.emplace_back(pr);
    return result;
}
