#include <bits/stdc++.h>

vector<vector<long long int>> printPascal(int n)
{
    vector<vector<long long int>> m;
    for (int line = 1; line <= n; line++)
    {
        vector<long long int> mid;
        long long int ans = 1;
        for (long long int i = 1; i <= line; i++)
        {
            mid.push_back(ans);
            ans = ans * (line - i) / i;
        }
        m.emplace_back(mid);
    }
    return m;
}
