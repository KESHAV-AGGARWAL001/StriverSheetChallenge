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
    
//     median in a stream
    
    
#include "bits/stdc++.h"
vector<int> findMedian(vector<int> &arr, int n){
	
	vector<int> ans;
	priority_queue<int> left;
	priority_queue<int, vector<int> , greater<int>> right;

	for(int i=0; i<n;i++){
		if(!left.empty() and left.top() > arr[i]){
			left.push(arr[i]);
			if(left.size() > right.size()+1) {
				right.push(left.top());
				left.pop();
			}
		}
		else{
			right.push(arr[i]);
			if(right.size() > left.size()+1){
				left.push(right.top());
				right.pop();
			}
		}

		if((i+1)&1){
			ans.push_back((left.size() > right.size()) ? left.top() : right.top());
		}
		else{
			ans.push_back((left.top()+right.top())/2);
		}
	}

	return ans;
	
}

    for(int i=n-1;i>=1;i--){
        result[i] = max(result[i], result[i+1]);
    }
    
    result.erase(result.begin());
    return result;
}



//  rod cutting problem 

int cutRod(vector<int> &price, int n)
{

	vector<int> dp(n+1,0);
	for(int i=1; i<=n;i++){
		int temp = 0;
		for(int j=0; j<i;j++){
			temp = max(temp, price[j] + dp[i-j-1]);
		}
		dp[i] = temp;
	}
	return dp[n];
}


//  cycle detection in undirected graph 
#include "bits/stdc++.h"

void dfs(int x, int p, vector<int>& col, vector<vector<int>>& v , bool &flag)
{
    col[x] = 1;
    for (auto itr : v[x])
    {
        if (itr == p)
        {
            continue;
        }
        if (col[itr] == 1)
        {
            flag = 1;
        }
        if (col[itr] == -1)
        {
            dfs(itr, x, col, v,flag);
        }
    }
    col[x] = 2;
}


string cycleDetection (vector<vector<int>>& edges, int n, int m)
{
	vector<vector<int>> adj(n+1);
    for(auto it : edges){
        int u = it[0];
        int v = it[1];
        adj[u].push_back(v);
        adj[v].push_back(u);
    } 
    vector<int>col(n + 1, -1);
	bool flag = false;
    for (int i = 1; i <= n; i++)
    {
        if (col[i] == -1) {
            dfs(i, 0, col, adj, flag);
        }
    }
    return flag == 1 ? "Yes" : "No";
}


// path in a tree
#include <bits/stdc++.h> 

void dfs(TreeNode<int> *root, vector<int> &ans, int x){
	if(!root) return;
	if(root->data == x){
		for(auto it: ans) cout<<it<<" ";
		cout<<x;
		return;
	}
	ans.push_back(root->data);
	dfs(root->left , ans, x);
	dfs(root->right, ans , x);
	ans.pop_back();
}

vector<int> pathInATree(TreeNode<int> *root, int x)
{
	vector<int>ans;
	dfs(root, ans , x );
	return {};
}


//  maximum product subarray 
//  I used idea of kadane algorithm 

#include <bits/stdc++.h> 
int maximumProduct(vector<int> &nums, int n)
{
	int maxi = INT_MIN;
	int prod=1;

	for(int i=0;i<n;i++)
	{
		prod*=nums[i];
		maxi=max(prod,maxi);
		if(prod==0)
		prod=1;
	}
	prod=1;
	for(int i=n-1;i>=0;i--)
	{
		prod*=nums[i];

		maxi=max(prod,maxi);
		if(prod==0)
		prod=1;
	}
	return maxi;
}


