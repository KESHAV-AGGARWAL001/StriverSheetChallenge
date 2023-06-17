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


//  maximum of every window size 
#include <bits/stdc++.h> 
vector<int> maxMinWindow(vector<int> a, int n) {
   
   stack<int> s;
   vector<int> prev_min(n+1);   
   vector<int> next_min(n+1);

   for(int i=0; i<n;i++) prev_min[i] = -1 , next_min[i] = n;
    
    for(int i=0; i<n;i++){
        while(!s.empty() and a[s.top()] >= a[i]){
            s.pop();
        }
        if(!s.empty()) prev_min[i] = s.top();
        s.push(i);
    }

    while(s.size()) s.pop();
    
    for(int i=n-1;i>=0;i--){
         while(!s.empty() and a[s.top()] >= a[i]){
            s.pop();
        }
        if(!s.empty()) next_min[i] = s.top();
        s.push(i);
    }

    while(s.size()) s.pop();
    
    vector<int> result (n+1, INT_MIN);

    for(int i=0; i<n;i++){
        int length = next_min[i] - prev_min[i] - 1;
        
        result[length] = max(result[length] , a[i]);
    }

    for(int i=n-1;i>=1;i--){
        result[i] = max(result[i], result[i+1]);
    }
    
    result.erase(result.begin());
    return result;
}
