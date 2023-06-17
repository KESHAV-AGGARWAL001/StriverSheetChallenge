// xor of subarray equal to x

#include <bits/stdc++.h>

int subarraysXor(vector<int> &arr, int x)
{
    unordered_map<int,int> ump;
    ump[0] =1;
    int count = 0;
    int xorProduct = 0;
    for(auto it : arr){
        xorProduct ^= it;
        if(ump[xorProduct ^ x] > 0) count+= ump[xorProduct^x];
        ump[xorProduct]++;
    }
    return count;
}
