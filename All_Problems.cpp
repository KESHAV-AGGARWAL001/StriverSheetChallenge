// Unique paths 

#include <bits/stdc++.h> 
int dfs(int startx, int starty , int m , int n, vector<vector<int>> &dp){
	if(startx == m-1 && starty == n-1) return 1;
	if(startx >= m || starty >= n) return 0;
	if(dp[startx][starty]!=-1) return dp[startx][starty];
	return dp[startx][starty] = dfs(startx+1, starty,m,n,dp) + dfs(startx, starty+1 , m,n,dp);
}
int uniquePaths(int m, int n) {
	vector<vector<int>> dp(m, vector<int> (n, -1));
    return dfs(0,0,m,n,dp);
}

// reverse pairs 
#include <bits/stdc++.h> 

int merge(int low , int high , int mid , vector<int> &arr){
	int cnt = 0;
	int j = mid+1;
	for(int i=low ; i<=mid;i++){
		while(j <= high and arr[i] > 2LL * arr[j]){
			j++;
		}
		cnt += (j-(mid+1));
	}

	vector<int> temp;
	int left = low , right = mid+1;
	while(left <= mid and right <= high){
		if(arr[left] <= arr[right]){
			temp.push_back(arr[left++]);
		}
		else {
			temp.push_back(arr[right++]);
		}
	}

	while(left <= mid){
		temp.push_back(arr[left++]);
	}
	while(right <= high ){
		temp.push_back(arr[right++]);
	}

	for(int m = low ; m<=high ; m++){
		arr[m] = temp[m-low];
	}
	return cnt;
}


int mergeSort(int low , int high , vector<int> &arr){
	if(low == high) return 0;
	int mid = (low+high)/2;
	int val = mergeSort(low , mid , arr)  + mergeSort(mid+1 , high , arr);
	val += merge(low, high , mid, arr);
	return val;
}
int reversePairs(vector<int> &arr, int n){
	return mergeSort(0,n-1 , arr);
}


// pair sum 

#include <bits/stdc++.h>

vector<vector<int>> pairSum(vector<int> &arr, int s){
   // Write your code here.
   unordered_map<int,int> mp;
   vector<vector<int>> ans;
   for(auto it : arr){
      int sum = s-it;
      if(mp.find(sum) != mp.end()){
         int count = mp[sum];
         for(int i=0 ;i<count;i++){
            ans.push_back({min(it, sum) , max(it,sum) });
         }
      }
      mp[it]++;
   }
   sort(ans.begin() , ans.end());
   return ans;
}

// find four Elements 
#include <bits/stdc++.h>

string fourSum(vector<int> nums, int target, int n) {
    sort(nums.begin(), nums.end());
    for(int i=0; i<n-3; i++){
        for(int j=i+1; j<n-2; j++){
            long long newTarget = (long long)target - (long long)nums[i] - (long long)nums[j];
            int low = j+1, high = n-1;
            while(low < high){
                if(nums[low] + nums[high] < newTarget){
                    low++;
                }
                else if(nums[low] + nums[high] > newTarget){
                    high--;
                }
                else{
                    return "Yes\n";
                }
            }
            while(j+1 < n && nums[j] == nums[j+1]) j++;
        }
        while(i+1 < n && nums[i] == nums[i+1]) i++;
    }
    return "No\n";
}


//  longest consecutive sequence 
#include <bits/stdc++.h>

int lengthOfLongestConsecutiveSequence(vector<int> &nums, int m) {
    if(nums.size()==0 || nums.size()==1) return nums.size();
    unordered_set<int> n;
    for(auto it:nums){n.insert(it);}
    vector<int> p;
    for(auto it:n) p.push_back(it);
    sort(p.begin(),p.end());
    int len = 0;
    int index = 0;
    int i;
    for(i=0; i<p.size()-1;i++){
        if(p[i]+1==p[i+1]){
            continue;
        }
        else{
            len = max(len, i-index+1);
            index = i+1;
        }
    }
    len = max(len,i-index+1);
    return len;
}

// longest subarray zero sum
#include <bits/stdc++.h>

int LongestSubsetWithZeroSum(vector < int > arr) {
  unordered_map<int,int> mp;
  int sum = 0;
  int max_size = 0;
  for(int i=0; i<arr.size(); i++){
    sum += arr[i];
    if(sum == 0) max_size = i+1;
    else if(mp.find(sum) != mp.end()){
      max_size = max(max_size, i - mp[sum] );
    }
    else{
      mp[sum] = i;
    }
  }
  return max_size;
}


// reverse a linked list

#include <bits/stdc++.h>
LinkedListNode<int> *reverseLinkedList(LinkedListNode<int> *head) 
{
    LinkedListNode<int>* prev = NULL;
    LinkedListNode<int>* curr = head;

    while(curr != NULL){
        LinkedListNode<int>* nextPtr = curr->next;
        curr->next = prev;
        prev = curr;
        curr = nextPtr;
        
    }
    return prev;
}


// longest subarray containing duplicate characters

#include <bits/stdc++.h> 
int uniqueSubstrings(string s)
{
    unordered_set<int> a;
    int maxlen = 0;
    for(int i=0; i<s.length();i++){
        int len =0;
        a.clear();
        for(int j=i ; j<s.length() ;j++){
            if(a.find(s[j])==a.end()){
                    a.insert(s[j]);
                len++;
            }
            else{
                break;
            }
        }
        maxlen = max(maxlen, len);
    }
    return maxlen;
}

// middle of linked list 

Node *findMiddle(Node *head) {
    if(head->next==NULL){
        return head;
    }
    int count =0 ;
    Node* p = head;
    while(p!=nullptr){
        count++;
        p = p->next;
    }
    if(count%2){
        Node *slow = head;
        count = count/2;
        while(count--){
            slow = slow->next;
        }
        return slow;
    }
    else {
        Node *fast = head;
        count = count/2;
        while(count--){
            fast = fast->next;
        }
        return fast;
    }
}

// merge sorted linked list 
#include <bits/stdc++.h>
Node<int>* sortTwoLists(Node<int>* first, Node<int>* second)
{
    Node<int> *result = new Node<int> (0);
    Node<int> *ans = result;
    
    while(first and second){
        if(first->data >= second->data){
            Node<int> *temp = new Node<int>(second->data);
            result->next = temp;
            result = result->next;
            second = second->next;
        }
        else {
            Node<int> *temp = new Node<int>(first->data);
            result->next = temp;
            result = result->next;
            first = first->next;
        }
    }
    
    if(first) result->next = first;
    if(second) result->next = second;

    return ans->next;
    
}

// delete kth node from ll

Node* removeKthNode(Node* head, int n)
{
    vector<int> a;
    Node* r = head;
    while(r!=nullptr){
        a.push_back(r->data);
        r = r->next;
    }
    int val = a.size() - n;
    Node* ans  = new Node(0);
    Node* ptr = ans;
    for(int i = 0;i<a.size() ;i++){
        if(i==val){
            continue;
        }
        else{
            ans->next = new Node(a[i]);
            ans = ans->next;
        }
    }
    return ptr->next;
}

//  add two numbers


Node *addTwoNumbers(Node *l1, Node *l2)
{
    Node* newList = new Node();
        Node* result = newList;
        int carry = 0;
        while(l1 and l2){
            int dataue = l1->data + l2->data;
            dataue += carry ;
            carry = (dataue>=10)?1:0;
            dataue = dataue%10;
            Node* temp = new Node(dataue);
            newList->next = temp;
            newList = newList->next;
            l1 = l1->next;
            l2=l2->next;
        }
        while(l1){
            int dataue = l1->data + carry;
            carry = (dataue>=10)?1:0;
            dataue = dataue%10;
            Node* temp = new Node(dataue);
            newList->next = temp;
            newList = newList->next;
            l1 = l1->next;
        }
        while(l2){
            int dataue = l2->data + carry;
            carry = (dataue>=10)?1:0;
            dataue = dataue%10;
            newList->next = new Node(dataue);;
            newList = newList->next;
            l2=l2->next;
        }
        if(carry == 1){
            newList->next = new Node(1);
        }
        return result->next;
}

// delete node in a ll

void deleteNode(LinkedListNode<int> * node) {
    LinkedListNode<int>* nex = new LinkedListNode<int>(node->next->data);
    LinkedListNode<int> * star = node;
    star->next = star->next->next;
    node->data = nex->data;
}

// cycle detection in ll

bool detectCycle(Node *head)
{
	Node *slow = head, *fast = head;
    while(fast!=nullptr && fast->next!=nullptr ){
        slow = slow->next ;
        fast = fast->next->next;
        if(slow == fast) return true;
    }
    return false;
}

//  palindrome ll 

bool isPalindrome(LinkedListNode<int> *head) {
    vector<int> temp;
    while(head){
        temp.push_back(head->data);
        head = head->next;
    }
    int i=0, j=temp.size()-1;
    while(i<j){
        if(temp[i]!=temp[j]) return false;
        i++ , j--;
    }
    return true;
}

// ll cycle 2 

Node *firstNode(Node *head)
{
    Node* slow = head;
    Node* fast = head;
    while (fast && fast->next) {
      slow = slow->next;
      fast = fast->next->next;
      if (slow == fast) {
        slow = head;
        while (slow != fast) {
          slow = slow->next;
          fast = fast->next;
        }
        return slow;
      }
    }
    return nullptr;
}

// rotate ll

Node *rotate(Node *head, int k) {
     vector<int> vec2;
     int size = 0;
     for(Node* temp = head ; temp!= nullptr ; temp = temp->next){
          vec2.push_back(temp->data);
          size++;
     }
     k = k%size;
     rotate(vec2.begin(), vec2.begin()+size-k, vec2.end());
     Node* result = new Node(0);
     Node* ans = result;
     for(auto it : vec2){
          Node* temp = new Node(it);
          result->next = temp;
          result = result->next;
     }
     return ans->next;
}

// copy ll with random pointer 
#include <bits/stdc++.h>
unordered_map<LinkedListNode<int>*, LinkedListNode<int>*> mp;
LinkedListNode<int> *cloneRandomList(LinkedListNode<int> *head)
{
    if (head == nullptr)
      return nullptr;
    if (mp.count(head))
      return mp[head];

    LinkedListNode<int>* newNode = new LinkedListNode<int>(head->data);
    mp[head] = newNode;
    newNode->next = cloneRandomList(head->next);
    newNode->random = cloneRandomList(head->random);
    return newNode;
}

// 3 sum 
#include <bits/stdc++.h> 
vector<vector<int>> findTriplets(vector<int>nums, int n, int K) {
	set<vector<int>> vec;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < nums.size() - 2; i++) {
			int left = i + 1, right = nums.size() - 1;
			while (left < right) {
				if (nums[i] + nums[left] + nums[right] < K) left++;
				else if (nums[i] + nums[left] + nums[right] > K) right--;
				else {
					vector<int> temp;
					temp.insert(temp.begin() , {nums[i] , nums[left] , nums[right]});
					vec.insert(temp);
					left++;right--;
				}
			}
        }
        vector<vector<int>> v;
        for(auto it:vec) v.push_back(it);
        return v;
}

// /trapping rainwater 
#include <bits/stdc++.h> 

long max(long one , long two ){
    return (one > two ) ? one : two;
}

long getTrappedWater(long *arr, int n){
    long ans = 0;

    int left = 0 , right = n-1; 
    long left_max = 0 , right_max = 0;

    while(left<=right){
        ((left_max >= right_max) ? (ans += max(0LL, right_max - arr[right]) , right_max = max(right_max, arr[right--])) :
            ( ans += max(0LL, left_max - arr[left]) , left_max = max(left_max, arr[left++])));
    }
    return ans;
}

// maximum consecutive ones 
int longestSubSeg(vector<int> &nums , int n, int k){
    int i =0 , j=0; 
    while(i<nums.size()){
        if(nums[i] ==0 ) k--;
        if(k<0){
            if(nums[j] == 0) k++;
            j++;
        }
        i++;
    }
    return i-j;
}
//  maximum meetings 
#include <bits/stdc++.h> 
vector<int> maximumMeetings(vector<int> &start, vector<int> &end) {
    vector<pair<int, int> > v;
    for(int i = 0 ; i<end.size() ;i++){
        v.push_back({end[i] , i});
    }

    sort(v.begin() , v.end());

    vector<int> ans;

    int prev = v[0].first;
    ans.push_back(v[0].second+1);

    for(int i=1 ; i<v.size(); i++){
        if(start[v[i].second] > prev){
            ans.push_back(v[i].second+1);
            prev = v[i].first;
        }
    }
    return ans;
}

//  job sequencing 
#include <bits/stdc++.h> 

struct profit{
    bool operator()(const vector<int> &a, const vector<int> &b){
        return a[1] < b[1];
    }
};
int jobScheduling(vector<vector<int>> &jobs)
{

    sort(jobs.begin() , jobs.end());

    int total_profit = 0;
    priority_queue<vector<int> , vector<vector<int>> , profit > pq;
    for(int i=jobs.size()-1 ; i>= 0; i--){
        int slots = 0;

        if(i==0) slots = jobs[i][0];
        else slots = jobs[i][0] - jobs[i-1][0];

        pq.push(jobs[i]);

        while(slots and pq.size()){
            auto it = pq.top();
            pq.pop();

            slots--;

            total_profit += it[1];
        }
    }
    return total_profit;
}


// fractional knapsack 
#include <bits/stdc++.h> 


static const bool comp(pair<int,int>& p1, pair<int,int> &p2)
{
    return ((1.0)*p2.first) / (p2.second) > ((1.0)*p1.first) / (p1.second);
}

double maximumValue (vector<pair<int, int>>& arr, int n, int w)
{
    double profit = 0;
    sort(arr.begin(), arr.end(), comp);
    
    for(int i = 0; i < n && w!=0; i++){
        profit += arr[i].second * (1.0 * min(arr[i].first, w) / arr[i].first);
        w -= min(arr[i].first, w);
    }
    
    return profit;
}

//  return subset sum to k 
#include "bits/stdc++.h"
void Recursion(int index , int n , int target , vector<int> &k , vector<int> nums , vector<vector<int>> &ans){
    if(index == n){
        int sum = accumulate(k.begin() , k.end() ,0);
        if(sum == target) ans.push_back(k); 
        return;
    }
    k.push_back(nums[index]);
    Recursion(index+1, n , target, k, nums ,ans);

    k.pop_back();
    Recursion(index+1,n,target,k,nums,ans);
}
vector<vector<int>> findSubsetsThatSumToK(vector<int> arr, int n, int target)
{
    vector<int> k;
    vector<vector<int>> ans;
    Recursion(0,n,target,k,arr,ans);
    return ans;
}

//  subset2 
#include <bits/stdc++.h> 
void Recursion(int index , int n , vector<int> &k , vector<int> nums , set<vector<int>> &ans){
    if(index == n){
        vector<int> temp = k;
        sort(temp.begin() , temp.end());
        ans.insert(temp);
        return;
    }
    k.push_back(nums[index]);
    Recursion(index+1, n, k, nums ,ans);

    k.pop_back();
    Recursion(index+1,n,k,nums,ans);
}
vector<vector<int>> uniqueSubsets(int n, vector<int> &nums)
{
    vector<int> k;
    set<vector<int>> ans;
    Recursion(0,n, k, nums , ans);
    vector<vector<int>> result ;
    for(auto it : ans) result.push_back(it);
    return result;
}

//  subset 
#include <bits/stdc++.h> 

void Recursion(int index , int n , int sum , vector<int> nums , vector<int>&ans){
    if(index == n){
        ans.push_back(sum);
        return;
    }
    sum += nums[index];
    Recursion(index+1, n, sum, nums,ans);

    sum -= nums[index];
    Recursion(index+1,n,sum,nums,ans);
}

vector<int> subsetSum(vector<int> &num)
{
    vector<int> ans;
    Recursion(0,num.size() , 0 , num, ans);
    sort(ans.begin() , ans.end());
    return ans;
}

// maximum activities 
#include "bits/stdc++.h"

int maximumActivities(vector<int> &start, vector<int> &finish) {
    vector<pair<int,int>> vec;
    for(int i=0; i<finish.size();i++){
        vec.push_back({finish[i] , start[i]});
    }
    sort(vec.begin() , vec.end());
    int count = 1;
    int prev = vec[0].first;
    for(int i=1;i<vec.size();i++){
        if(vec[i].second >= prev){
            count++;
            prev = vec[i].first;
        }
    }
    return count;
}

// combination sum 

#include "bits/stdc++.h"
void Recursion(int index, int sum ,vector<int> k, vector<int> array, int target , vector<vector<int>> &vec)
{
    if (sum == target)
    {
        vec.push_back(k);
        return;
    }
    if(index==array.size() || sum > target) return;
    k.push_back(array[index]);
    Recursion(index+1 , sum += array[index] , k, array, target , vec);

    k.pop_back();
    while(index < array.size()-1 && array[index] == array[index + 1])
            index++;
    Recursion(index + 1, sum -= array[index] , k, array, target,vec);

}

vector<vector<int>> combinationSum2(vector<int> &candidates, int n, int target)
{
	vector<int> k;
    sort(candidates.begin(), candidates.end());
    vector<vector<int>> ans;
    Recursion(0,0,k,candidates,target,ans);
    return ans;
}

// print permutations 
#include <bits/stdc++.h> 


void permutation(string &s , int left , int right, vector<string> &st){
    if(left == right){
        st.push_back(s);
        return;
    }

    for(int i=left ; i <= right;i++){
        swap(s[left] , s[i]);
        permutation(s , left+1, right , st);
        swap(s[left] , s[i]);
    }
}
vector<string> findPermutations(string &s) {
    vector<string> st;
    int left = 0 , right = s.length()-1;
    permutation(s, left, right , st);
    return st;
}

// valid sudoku 
bool isValid(int board[9][9], int row, int col , int c){
    for(int i=0; i<9;i++){
    if(board[i][col] == c or board[row][i] == c or board[3*(row/3)+(i/3)][3*(col/3)+(i%3)] == c) return false;
    }
    return true;
}
bool solve(int board[9][9]){
    for(int i=0;i<9;i++){
        for(int j=0; j<9;j++){
            if(board[i][j] == 0){
                for(int c = 1 ; c<= 9 ;c++){
                    if(isValid(board,i,j,c)){
                    board[i][j] = c;
                    if(solve(board)) return true;
                    else board[i][j] = 0;
                    }
                }
            return false;
            }
        }
    }
    return true;
}


bool isItSudoku(int board[9][9]) {
    return solve(board);
}

// rat in a maze 
#include <bits/stdc++.h> 

void Recursion(int i, int j , int n , vector<vector<int>> maze , vector<vector<bool>> &visited , vector<vector<int>> &temp , vector<vector<int>>& ans){
  if(i == n-1 and j == n-1) {
     temp[i][j] = true;
     vector<int> result;
     for(int i=0; i<n;i++){
       for(int j=0; j<n;j++){
         result.push_back(temp[i][j]);
       }
     }
     ans.push_back(result);
     return ;
  }


    if (i + 1 < n and !visited[i + 1][j] && maze[i + 1][j] == 1) {
      visited[i][j] = 1;
      temp[i][j] = 1;
      Recursion(i+1, j,  n, maze, visited, temp, ans);
      visited[i][j] = 0;
      temp[i][j] = 0;
    }

    if (j - 1 >= 0 && !visited[i][j - 1] && maze[i][j - 1] == 1) {
      visited[i][j] = 1;
      temp[i][j] = 1;
      Recursion(i, j-1,  n,  maze, visited, temp, ans);
      visited[i][j] = 0;
      temp[i][j] = 0;
    }

    if (j + 1 < n && !visited[i][j + 1] && maze[i][j + 1] == 1) {
      visited[i][j] = 1;
      temp[i][j] = 1;
      Recursion(i, j+1,  n, maze, visited, temp, ans);
      visited[i][j] = 0;
      temp[i][j] = 0;
    }

    if (i - 1 >= 0 && !visited[i-1][j] && maze[i-1][j] == 1) {
      visited[i][j] = 1;
      temp[i][j] = 1;
      Recursion(i-1, j,  n, maze, visited, temp, ans);
      visited[i][j] = 0;
      temp[i][j] = 0;
    }
}
vector<vector<int> > ratInAMaze(vector<vector<int> > &maze, int n){
    vector<vector<int>> ans;
    if(maze[0][0] != 1 or maze[n-1][n-1] != 1) return ans;
    vector<vector<int>> temp(n, vector<int> (n,0));
    vector<vector<bool>> visited(n, vector<bool> (n, false));
    Recursion(0,0,n, maze,visited, temp,ans);
    return ans;
}

//  word break 2
#include <bits/stdc++.h> 
vector<string> wordBreak(string &s, vector<string> &wordDict)
{
    int n=s.size();
    unordered_set<string>word_Set(wordDict.begin(),wordDict.end());
    vector<vector<string>>dp(n+1,vector<string>());
    dp[0].push_back("");

    for(int i = 0; i < n; ++i){
        for(int j = i+1; j <= n; ++j){
            string temp = s.substr(i, j-i);
            if(word_Set.count(temp)){
                for(auto x : dp[i]){
                    dp[j].emplace_back(x + (x == "" ? "" : " ") + temp);  
                }
            }
        }
    }
    return dp[n];
}

//  matrix median 
#include "bits/stdc++.h"
int getMedian(vector<vector<int>> &matrix)
{
    vector<int> ans;
    for(auto it : matrix){
        for(auto m : it){
            ans.push_back(m);
        }
    }
    sort(ans.begin() , ans.end());
    return (ans.size() & 1) ? ans[ans.size()/2] : (ans[ans.size()/2] + ans[ans.size()/2 + 1])/2;
}

// /search in a rotated sorted array 
#include "bits/stdc++.h"
int search(int* nums, int n, int target) {
    int left = 0;
    int right = n-1;
    while(left <= right){
        int mid = left + (right - left) / 2;    
        if(nums[mid] == target)
            return mid;
        if(nums[mid] >= nums[left]) {
            if(target >= nums[left] && target <= nums[mid])
            {
                right = mid - 1;
            }
            else left = mid + 1;
        } 
        else {
            if(target >= nums[mid] && target <= nums[right]) 
            left = mid + 1;
            else right = mid - 1;
        }
    }
    return -1;
}

// median of two sorted arrays 
double median(vector<int>& nums1, vector<int>& nums2) {
	int n1=nums1.size();
        int n2=nums2.size();
        int n=n1+n2;
         
      if(n1>n2)  return median(nums2,nums1);
        int partition=(n+1)/2; 
        
    
    if(n1==0)
        return n2%2?nums2[n2/2]:(nums2[n2/2]+nums2[n2/2-1])/2.0;
    
    if(n2==0)
        return n1%2?nums1[n1/2]:(nums1[n1/2]+nums1[n1/2-1])/2.0;
    
    int left1=0;
    int right1=n1;
    int cut1,cut2;
    int l1,r1,l2,r2;
    
    do
    {   
        cut1=(left1+right1)/2;
        cut2=partition-cut1;
   
        l1=cut1==0?INT_MIN:nums1[cut1-1];
        
        l2=cut2==0?INT_MIN:nums2[cut2-1];
        
        r1=cut1>=n1?INT_MAX:nums1[cut1];
        
        r2=cut2>=n2?INT_MAX:nums2[cut2];
        
        if(l1<=r2&&l2<=r1)
             return n%2?max(l1,l2):(max(l1,l2)+min(r1,r2))/2.0;
        else
            
        if(l1>r2)
            right1=cut1-1;
        else
             left1=cut1+1;
       
       
    }while(left1<=right1);
        
             
    return 0.0;
}

// min heap

#include <bits/stdc++.h> 
vector<int> minHeap(int n, vector<vector<int>>& q) {
    priority_queue<int, vector<int> , greater<int> > pq;
    vector<int> ans;
    for(auto it : q){
        if(it[0] == 0){
            pq.push(it[1]);
        }
        if(it[0] != 0){
            ans.push_back(pq.top());
            pq.pop();
        }
    }
    return ans;
}

// kth smallest and largest 
#include <bits/stdc++.h>

vector<int> kthSmallLarge(vector<int> &arr, int n, int k)
{
	sort(arr.begin() , arr.end());
	vector<int> ans = {arr[k-1] , arr[n-k]};
	return ans;
}

// k max sum combinations 
#include <bits/stdc++.h> 
vector<int> kMaxSumCombination(vector<int> &a, vector<int> &b, int n, int k){
	vector<int> ans;
	for(int i=0; i<n;i++){
		for(int j=0; j<n;j++){
			ans.push_back(a[i] + b[j]);
		}
	}
	sort(ans.begin() , ans.end());
	vector<int> result;
	for(int i=0; i<k;i++){
		result.push_back(ans.back());
		ans.pop_back();
	}
	return result;

}

// running median
#include "bits/stdc++.h"


void findMedian(int *arr, int n)
{
    priority_queue<int> left;
    priority_queue<int, vector<int>, greater<int> > right;
    for(int i=0;i<n;i++){
        if (!left.empty() && left.top() > arr[i]) {
            left.push(arr[i]);
            if (left.size() > right.size() +1){
                right.push(left.top()); 
                left.pop();
            }
        }
        else {
            right.push(arr[i]);
            if (right.size() > left.size() +1){
                left.push(right.top()); 
                right.pop();
            }
        }
        if (i+1 & 1) {
            int median = left.size() > right.size() ? left.top() : right.top();
            cout<<median<<" ";
        } else {
          int median = (left.top() + right.top()) / 2;
          cout << median << " ";
        }
    }
}

//  merge k sorted arrays 
#include <bits/stdc++.h> 
vector<int> mergeKSortedArrays(vector<vector<int>>&kArrays, int k)
{
    vector<int> ans;
    for(auto it : kArrays){
        for(auto m : it) ans.push_back(m);
    }
    sort(ans.begin() , ans.end());
    return ans;
}

// k most frequent 
#include <bits/stdc++.h> 
vector<int> KMostFrequent(int n, int k, vector<int> &nums)
{
    unordered_map<int, int> m;
    for (auto it : nums)
    {
        m[it]++;
    }
    vector<pair<int, int>> pr;
    for (auto it : m)
    {
        pr.push_back({it.second, it.first});
    }
    sort(pr.begin(), pr.end());
    reverse(pr.begin(), pr.end());
    auto it = pr.begin();
    vector<int>ans;
    for(int i= 0;i<k ;i++){
        ans.push_back(it->second);
        it++;
    }
    sort(ans.begin() , ans.end());
    return ans;
}

// number of distinct substrings 
#include <bits/stdc++.h> 
int distinctSubstring(string &word) {
    //  Write your code here.
    unordered_set<string> ans;
    string s = "";
    for(int i=0;i<word.length();i++){
        for(int j=i; j<word.length() ;j++){
            s += word[j];
            ans.insert(s);
        }
        s.clear();
    }
    return ans.size();
}


// next greater element 
#include <bits/stdc++.h> 

vector<int> nextGreater(vector<int> &arr, int n) {
    stack<pair<int,int>> s;
    vector<int> ans(n);
    s.push({arr[0] , 0});

    for (int i = 1; i < n; i++) {

        if (s.empty()) {
            s.push({arr[i] , i});
            continue;
        }

        while (s.empty() == false && s.top().first < arr[i]) {
            ans[s.top().second] = arr[i];
            s.pop();
        }

        s.push({arr[i] , i});
    }

    while (s.empty() == false) {
        ans[s.top().second]  = -1;
        s.pop();
    }
    return ans;
}


//.valid parenthesis
bool isValidParenthesis(string s)
{
    stack<char> ans;
        bool flag = true;
        if(s.length()&1){
            return false;
        }
        for(int i=0; i<s.length() ;i++){
            if(s[i] == '(' || s[i] == '['||s[i] == '{'){
                ans.push(s[i]);
            }
            else if(ans.empty()){
                return false;
            }
            else if(ans.top() == '(' && s[i] == ')'){
                ans.pop();
            }
            else if(ans.top() == '[' && s[i] == ']'){
                ans.pop();
            }
            else if(ans.top() == '{' && s[i] == '}'){
                ans.pop();
            }
            else{
                return false;
            }
        }
        return ans.empty();
}

// queue using stack 
#include "bits/stdc++.h"
class Queue {
    // Define the data members(if any) here.
    public:
    stack<int> pq;
    Queue() {
        // Initialize your data structure here.
    }

    void enQueue(int val) {
        // Implement the enqueue() function.
        pq.push(val);
    }

    int deQueue() {
        if(pq.size() == 0) return -1;
        stack<int> temp;
        while(pq.size()){
            temp.push(pq.top());
            pq.pop();
        }
        int val = temp.top();
        temp.pop();
        while(temp.size()){
            pq.push(temp.top());
            temp.pop();
        }
        return val;
        // Implement the dequeue() function.
    }

    int peek() {
        if(pq.size() == 0) return -1;
        stack<int> temp;
        while(pq.size()){
            temp.push(pq.top());
            pq.pop();
        }
        int val = temp.top();
        while(temp.size()){
            pq.push(temp.top());
            temp.pop();
        }
        return val;
        // Implement the peek() function here.
    }

    bool isEmpty() {
        return pq.size() == 0;
        // Implement the isEmpty() function here.
    }
};

// stack using queue
#include <bits/stdc++.h> 
class Stack {
	// Define the data members.
    deque<int> st;
   public:
    Stack() {
        // Implement the Constructor.
    }

    /*----------------- Public Functions of Stack -----------------*/

    int getSize() {
        // Implement the getSize() function.
        return st.size();
    }

    bool isEmpty() {
        // Implement the isEmpty() function.
        return st.size()==0;
    }

    void push(int element) {
        // Implement the push() function.
        st.push_back(element);
    }

    int pop() {
        // Implement the pop() function.
        if(st.size() == 0) return -1;
        int val = st.back();
        st.pop_back();
        return val;
    }

    int top() {
        if(st.size() == 0) return -1;
        return st.back();
        // Implement the top() function.
    }
};

// implement queue
#include <bits/stdc++.h> 
class Queue {
public:
    deque<int> q;
    Queue() {
        // Implement the Constructor
    }

    bool isEmpty() {
        return q.size() == 0;
        // Implement the isEmpty() function
    }

    void enqueue(int data) {
        q.push_back(data);
        // Implement the enqueue() function
    }

    int dequeue() {
        if(q.size() == 0) return -1;
        int val = q.front();
        q.pop_front();
        return val;
        // Implement the dequeue() function
    }

    int front() {
        if(q.size() == 0) return -1;
        return q.front();
        // Implement the front() function
    }
};




// stack implement 
#include <bits/stdc++.h> 
// Stack class.
class Stack {
    vector<int> ans;
    int max_size ;
public:
    
    Stack(int capacity) {
        // Write your code here.
        max_size = capacity;
    }

    void push(int num) {
        // Write your code here.
        if(ans.size() != max_size) ans.push_back(num);
    }

    int pop() {
        // Write your code here.
        if(ans.size() == 0) return -1;
        int val = ans.back();
        ans.pop_back();
        return val;
    }
    
    int top() {
        // Write your code here.
        return (ans.size()!=0) ? ans.back() : -1;
    }
    
    int isEmpty() {
        // Write your code here.
        return ans.empty();
    }
    
    int isFull() {
        return ans.size() == max_size ? 1 :0;
        // Write your code here.
    }
    
};

// power set 
#include <bits/stdc++.h> 
void Recursion(int index , int n , vector<int> &k , vector<int> nums , vector<vector<int>> &ans){
    if(index == n){
        ans.push_back(k);
        return;
    }
    k.push_back(nums[index]);
    Recursion(index+1, n, k, nums ,ans);

    k.pop_back();
    Recursion(index+1,n,k,nums, ans);
}
vector<vector<int>> pwset(vector<int> nums)
{
    vector<int> k;
    vector<vector<int>> ans;
    Recursion(0,nums.size() , k, nums , ans);
    return ans;
}


//  valid parenthesis
bool isValidParenthesis(string s)
{
    stack<char> ans;
        bool flag = true;
        if(s.length()&1){
            return false;
        }
        for(int i=0; i<s.length() ;i++){
            if(s[i] == '(' || s[i] == '['||s[i] == '{'){
                ans.push(s[i]);
            }
            else if(ans.empty()){
                return false;
            }
            else if(ans.top() == '(' && s[i] == ')'){
                ans.pop();
            }
            else if(ans.top() == '[' && s[i] == ']'){
                ans.pop();
            }
            else if(ans.top() == '{' && s[i] == '}'){
                ans.pop();
            }
            else{
                return false;
            }
        }
        return ans.empty();
}

// next greater element

#include <bits/stdc++.h> 

vector<int> nextGreater(vector<int> &arr, int n) {
    stack<pair<int,int>> s;
    vector<int> ans(n);
    s.push({arr[0] , 0});

    for (int i = 1; i < n; i++) {

        if (s.empty()) {
            s.push({arr[i] , i});
            continue;
        }

        while (s.empty() == false && s.top().first < arr[i]) {
            ans[s.top().second] = arr[i];
            s.pop();
        }

        s.push({arr[i] , i});
    }

    while (s.empty() == false) {
        ans[s.top().second]  = -1;
        s.pop();
    }
    return ans;
}

// sort a stack
#include <bits/stdc++.h> 
void sortStack(stack<int> &s)
{
	stack<int> temp;

	while(!s.empty()){

		int value = s.top();
		s.pop();

		while(!temp.empty() and temp.top() < value){
			s.push(temp.top());
			temp.pop();
		}

		temp.push(value);
	}

	while(temp.empty() == false){
		s.push(temp.top());
		temp.pop();
	}
}

// next smaller element 
#include "bits/stdc++.h"

vector<int> nextSmallerElement(vector<int> &arr, int n)
{
    stack<pair<int,int>> s;
    vector<int> ans(n);
    s.push({arr[0] , 0});

    for (int i = 1; i < n; i++) {

        if (s.empty()) {
            s.push({arr[i] , i});
            continue;
        }

        while (s.empty() == false && s.top().first > arr[i]) {
            ans[s.top().second] = arr[i];
            s.pop();
        }

        s.push({arr[i] , i});
    }

    while (s.empty() == false) {
        ans[s.top().second]  = -1;
        s.pop();
    }
    return ans;
}


// lru cache
#include "bits/stdc++.h"

class LRUCache
{
public:
    unordered_map<int, int>m;
    unordered_map<int, int>visited;
    deque<int>q;
    int count = 0;
    int capacity = 0;
    LRUCache(int capacity)
    {
        // Write your code here
        this->capacity = capacity;
    }

    int get(int key)
    {
        // Write your code here
        if(!m.count(key) || m[key] == -1) return -1;
        q.push_back(key);
        visited[key]++;
        return m[key];
    }

    void put(int key, int value)
    {
        // Write your code here
        if(!m.count(key)|| m[key] == -1) count++;
        else visited[key]++;
        if(count > capacity){
            while(visited[q.front()]) visited[q.front()]--, q.pop_front();
            m[q.front()] = -1;
            q.pop_front();
            count--;
        }
        q.push_back(key);
        m[key] = value;
    }
};


//  largest rectangle 
 #include "bits/stdc++.h"
 
 int largestRectangle(vector < int > & heights) {
    stack<int>stk;
        int i=0;
        int n=heights.size();
        int maxArea = 0;
        while(i < n)
            {   

    if(stk.empty() or( heights[stk.top()] <= heights[i] ))
            {
                stk.push(i++);
            }
            else
            {
                int minBarIndex = stk.top();
                stk.pop();
                int width = i;
                if(!stk.empty())  width = i - 1 - stk.top();
                int area = heights[minBarIndex] * width;
                maxArea = max(maxArea , area);
            }
        }

        while(!stk.empty())
        {
            int minBarIndex = stk.top();
            stk.pop();
            int width = i;
            if(!stk.empty())  width = i - 1 - stk.top();
            int area = heights[minBarIndex] * width;
            maxArea = max(maxArea , area);
        }
        return maxArea;
 }

//  sliding window maximum 
#include <bits/stdc++.h> 
vector<int> slidingWindowMaximum(vector<int> &nums, int &k)
{
    vector<int> ans;
    deque<int> dq;
    int i;
    for( i=0; i<k;i++){
        while(dq.size() && nums[i] >= nums[dq.back()]) dq.pop_back();
        dq.push_back(i);
    }
    for( ; i<nums.size();i++){
        ans.push_back(nums[dq.front()]);
        while(dq.size() && dq.front() <= i-k) dq.pop_front();
        while(dq.size() && nums[i] >= nums[dq.back()]) dq.pop_back();
        dq.push_back(i);
    }
    ans.push_back(nums[dq.front()]);
    return ans;
}

//  min stack
#include <bits/stdc++.h> 
// Implement class for minStack.
class minStack
{
	// Write your code here.
	
	public:
		 stack<pair<int, int> > s;
		// Constructor
		minStack() 
		{ 
			// Write your code here.
		}
		
		// Function to add another element equal to num at the top of stack.
		void push(int element)
		{
			int new_min = s.empty()? element : min(element, s.top().second); 
        	s.push({ element, new_min });
		}
		
		// Function to remove the top element of the stack.
		int pop()
		{
			int popped;
        	if (!s.empty()) {
				popped = s.top().first;
				s.pop();
				return popped;
			}
			return -1;
		}
		
		// Function to return the top element of stack if it is present. Otherwise return -1.
		int top()
		{

			if(s.size() == 0) return -1;
			return s.top().first;
			
		}
		
		// Function to return minimum element of stack if it is present. Otherwise return -1.
		int getMin()
		{
			if(s.size() == 0) return -1;
			int min_elem = s.top().second;
        	return min_elem;
		}
};

//  rotting oranges
 #include "bits/stdc++.h"

int minTimeToRot(vector<vector<int>>& grid, int n, int m)
{
    vector<vector<bool>> visited(n,vector<bool> (m,false));
    queue<pair<pair<int,int> , int>> q;
    for(int i=0; i<n;i++){
        for(int j=0; j<m;j++){
        if(grid[i][j] == 2){
            visited[i][j] = true;
            q.push({{i,j},0});
        }
        }
    }
    int min_time = 0;
    while(q.size()){
        int r = q.front().first.first;
        int c = q.front().first.second;
        int time = q.front().second;
        min_time = time;
        q.pop();
        int row[] = {-1,1,0,0};
        int col[] = {0,0,-1,1};

        for(int i=0; i<4;i++){
        int ro = r + row[i];
        int co = c + col[i];
        if(ro<n and ro>=0 and co>=0 and co<m ){
            if(!visited[ro][co] and grid[ro][co] == 1){
            grid[ro][co] = 2;
            visited[ro][co] = true;
            q.push({{ro,co},time+1});
            }
        }
        }
    }
    for(int i=0; i<n;i++){
        for(int j=0; j<m;j++){
        if(grid[i][j] == 1) return -1;
        }
    }
    return min_time; 
}

//  maximum of minimum of all size 

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

//  reverse words in a string 
string reverseString(string &str){
	vector<string> ans;
	string temp = "";
	for(int i=0; i<str.length();i++){
		if(str[i] == ' '){
			ans.push_back(temp);
			temp = "";
			continue;
		}
		temp += str[i];
	}
	ans.push_back(temp);
	reverse(ans.begin() , ans.end());
	string result = "";
	for(auto it : ans){
		if(it.length()) {
			result += it + " ";
		}
	}
	result.pop_back();
	return result;
}

// roman numerals to integers 
#include <bits/stdc++.h> 
int romanToInt(string str) {
    map<char, int> a ;
        a.insert({'I' , 1});
        a.insert({'V' , 5});
        a.insert({'X' , 10});
        a.insert({'L' , 50});
        a.insert({'C' , 100});
        a.insert({'D' , 500});
        a.insert({'M' , 1000});
        int sum = 0;
    for (int i = 0; i < str.length(); i++) 
    {
        if (a[str[i]] < a[str[i + 1]])
        {
            sum+=a[str[i+1]]-a[str[i]];
            i++;
            continue;
        }
        sum += a[str[i]];
    }
    return sum;
}

//  implement trie 
#include "bits/stdc++.h"
struct Node{
    Node* child[26];
    bool isEnd;

    Node(){
        memset(child , 0, sizeof child);
        isEnd = false;
    }
};

class Trie {
    Node* root;
public:

    Trie() {
        root = new Node();
    }

    /** Inserts a word into the trie. */
    void insert(string word) {
        Node* temp = root;
        for(auto it : word){
            int index = it-'a';
            if(temp->child[index] == NULL){
                temp->child[index] = new Node();
            }
            temp = temp->child[index];
        }
        temp->isEnd = true;
    }

    /** Returns if the word is in the trie. */
    bool search(string word) {
        Node* temp = root;
        for(auto it : word){
            int index = it-'a';
            if(temp->child[index] == NULL) return false;
            temp = temp->child[index];
        }
        return temp->isEnd;
    }

    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        Node* temp = root;
        for(auto it : prefix){
            int index = it-'a';
            if(temp->child[index] == NULL) return false;
            temp = temp->child[index];
        }
        return true;
    }
};

//  implement trie 2
#include <bits/stdc++.h> 

struct Node{
    Node* child[26];
    int countWords;
    int isEnd;

    Node(){
        memset(child , 0 , sizeof child);
        countWords = 0;
        isEnd = 0;
    }
};
class Trie{
    private :
        Node* root;
    public:

    Trie(){
        root = new Node();
    }

    void insert(string &word){
        Node* temp = root;
        for(auto it : word){
            int index = it-'a';
            if(temp->child[index] == NULL){
                temp->child[index] = new Node();
            }
            temp->countWords++;
            temp = temp->child[index];
        }
        temp->isEnd++;
        temp->countWords++;
    }

    int countWordsEqualTo(string &word){
        Node* temp = root;
        for(auto it : word){
            int index = it-'a';
            if(temp->child[index] == NULL) return 0;
            temp = temp->child[index];
        }
        return temp->isEnd;
    }

    int countWordsStartingWith(string &word){
        Node* temp = root;
        for(auto it : word){
            int index = it-'a';
            if(temp->child[index] == NULL) return 0;
            temp = temp->child[index];
        }
        return temp->countWords;
    }

    void erase(string &word){
        Node* temp = root;
        for(auto it : word){
            int index = it-'a';
            temp->countWords--;
            temp = temp->child[index];
        }
        temp->countWords--;
        temp->isEnd--;
    }
};


// intersection of two linked list 

#include "bits/stdc++.h"
Node* findIntersection(Node *firstHead, Node *secondHead)
{
    unordered_set<Node* > ump;
    
    for(Node* temp = firstHead ; temp!=NULL; temp = temp->next){
        ump.insert(temp);
    }

    for(Node* temp = secondHead ;temp!=NULL;temp = temp->next){
        if(ump.find(temp) != ump.end()) return temp;
    }

    return NULL;
    
}

//  remove duplicates from sorted array 
int removeDuplicates(vector<int> &arr, int n) {
	int temp = arr[0];
	int count = 1;
	for(int i=1; i<arr.size();i++){
		if(temp!=arr[i]){
			temp = arr[i];
			count++;
		}
	}
	return count;
}


//  complete string (trie problem) 

#include <bits/stdc++.h>

struct Node{
    Node* child[26];
    bool isEnd;

    Node(){
        memset(child , 0, sizeof child);
        isEnd = false;
    }
};

class Trie {
    Node* root;
public:

    Trie() {
        root = new Node();
    }
    void insert(string word) {
        Node* temp = root;
        for(auto it : word){
            int index = it-'a';
            if (temp->child[index] == NULL) {
              temp->child[index] = new Node();
            }
            temp = temp->child[index];
        }
        temp->isEnd = true;
    }

    bool search(string word){
        Node* temp = root;
        for(auto it : word){
            int index = it-'a';
            if(temp->child[index]->isEnd == false) return false;
            temp = temp->child[index];
        }
        return true;
    }
};

string completeString(int n, vector<string> &a) {
    Trie* tr = new Trie();
    for(auto it : a){
        tr->insert(it);
    }
    string result = "";
    for(auto it: a){
        if(tr->search(it)){
            if(it.length() == result.length() and lexicographical_compare(it.begin() , it.end() , result.begin() , result.end())) result = it;
            else if(it.length() > result.length()) result = it;
        }
    }
    return result.length() ? result : "None";
}


// N-Queens

void search(int y, vector<bool> &column , vector<bool > &diag1 ,  vector<bool> &diag2 , vector<vector<int>> &board ,  int n  , vector<vector<vector<int>>>& ans)
{
    if (y == n)
    {
        ans.push_back(board);
        return;
    }
    for (int x = 0; x < n; x++)
    {
        if (column[x] || diag1[x + y] || diag2[x - y + n - 1]) continue;
        column[x] = diag1[x + y] = diag2[x - y + n - 1] = true;
        board[x][y] = 1;
        search(y + 1 , column , diag1, diag2 , board, n , ans);
        board[x][y] = 0;
        column[x] = diag1[x + y] = diag2[x - y + n - 1] = false;
    }
}

vector<vector<int>> solveNQueens(int n) {
    vector<vector<vector<int>>> ans;
    vector<vector<int>> board(n , vector<int> (n,0));
    vector<bool> column(n) , diag1(2*n-1), diag2(2*n-1);
    search(0, column,diag1, diag2 , board , n  , ans);
    for(auto it : ans){
        for(int i=0; i<n;i++){
            for(int j=0; j<n;j++){
                cout<<it[i][j]<<" ";
            }
        }
        cout<<endl;
    }
}

//  Single Element in an array 

int singleNonDuplicate(vector<int>& nums)
{
	int l=0;
	int r=nums.size()-1;
	while(l<r){
		int mid=(l+r)/2;
		if(mid%2==0){
			if(nums[mid]==nums[mid+1]){
				l=mid+1;
			}
			else{
				r=mid;
			}
		}
		else{
			if(nums[mid]!=nums[mid+1]){
				l=mid+1;
			}
			else{
				r=mid;
			}
		}
	}
	return nums[l];
}

// / invert a tree 

#include <bits/stdc++.h> 

bool extract_path(TreeNode<int> *root, TreeNode<int> *leaf, stack<TreeNode<int>*> &path) {
    if(root == NULL) return 0;

    path.push(root);

    if(root->data == leaf->data) return 1;
    bool lf = extract_path(root->left, leaf, path);
    bool rg = extract_path(root->right, leaf, path);

    if(lf || rg) return 1;

    path.pop();
    return 0;
}

TreeNode<int> * invertBinaryTree(TreeNode<int> *root, TreeNode<int> *leaf)
{
	// Write your code here.
    if(root == NULL or leaf == NULL) return NULL;
    
    stack<TreeNode<int>*> path;
    extract_path(root, leaf, path);


    TreeNode<int> *parent = path.top();
    path.pop();

    TreeNode<int> *new_root = parent;

    while(!path.empty()) {
        TreeNode<int>* cur = path.top();
        path.pop();

        if(cur->left == parent) {
            cur->left = NULL;
        }

        else {
            cur->right = cur->left;
            cur->left = NULL;
        }

        parent->left = cur;

        parent = cur;
    }

    return new_root;
}


//  implement atoi function 
#include <bits/stdc++.h> 
int atoi(string str) {
    int len = str.length();
    string res = "";
    for(int i=1; i<len;i++){
        if(str[i] >= '0' and str[i]<='9'){
            res += str[i];
        }
    }
    if((str[0] == '-' or ( str[0] >= '0' and str[0] <= '9') )and res.length() ){
        res = str[0] + res;
    }
    return res.length() > 0 ? stoi(res) : 0;
}

// longest common prefix 

#include "bits/stdc++.h"

string longestCommonPrefix(vector<string> &strs, int n)
{
    int len = strs[0].length();
    string s = strs[0];
    for(int i=1;i<n;i++){
        if(strs[i].length()<len){
            len = strs[i].length();
            s = strs[i];
        }
    }
    string a = "";
    for(int i=0;i<len;i++){
        bool flag =0 ;
        for(int j=0; j<n ; j++){
            if(strs[j][i] == s[i]){
                flag = 1;
            }
            else if(strs[j][i] != s[i]){
                flag = 0;
                break;
            }
        }
        if(flag){
            a += s[i];
        }
        else{
            break;
        }
    }
    return a;
}



// search pattern 
vector<int> stringMatch(string &str, string &pat) {
    vector<int> index;
    if(str == pat) return {0};
    if(str.length() < pat.length()) return index;
    for(int i=0 ; i<=str.length() - pat.length() ; i++){
        if(str[i] == pat[0] and str.substr(i,pat.length()) == pat) {
            index.push_back(i);
        }
    }
    return index;
}


// z - algorithm,
#include "bits/stdc++.h"

int zAlgorithm(string s, string p, int n, int m)
{
	int count = 0;
	for(int i=0;i<=s.length() - p.length() ;i++){
		if(s[i] == p[0] and s.substr(i,p.length()) == p){
			count++;
		}
	}
	return count;
}

// find pattern 

#include <bits/stdc++.h> 
bool findPattern(string p, string s)
{
    return (s.find(p) != string::npos);
}

//  check permutations 
#include <bits/stdc++.h> 
bool areAnagram(string &str1, string &str2){
    int count[26] = {0};
    if(str1.length() != str2.length()) return false;
    for(auto it : str1){
        count[it-'a']++;
    }
    for(auto it : str2){
        count[it-'a']--;
    }
    for(int i=0; i<26;i++){
        if(count[i] != 0) return false;
    }
    return true;
}

//  count and say 
#include <bits/stdc++.h> 
string writeAsYouSpeak(int n) 
{
	
	 string s = "1";
	if(n==1) return s;
	n--;
	while(n--){
		string temp = "";
		int freq = 1;
		char ch = s[0];
		for(int i=1; i<s.length() ;i++){
			if(ch == s[i]){
				freq++;
			}
			else {
				temp += to_string(freq);
				temp += ch;
				ch = s[i];
				freq = 1;
			}
		}
		temp += to_string(freq);
		temp += ch;
		s = temp;
	}
	return s;
}

// inorder traversal 
#include <bits/stdc++.h> 

void inorder(TreeNode* root, vector<int>&v){
    if(root==nullptr) return ;
    inorder(root->left, v);
    v.push_back(root->data);
    inorder(root->right,v);
}
vector<int> getInOrderTraversal(TreeNode *root)
{
    vector<int> v;
    inorder(root , v);
    return v;
}

//  preorder traversal 
#include <bits/stdc++.h> 
/*
    Following is Binary Tree Node structure:
    class TreeNode
    {
    public:
        int data;
        TreeNode *left, *right;
        TreeNode() : data(0), left(NULL), right(NULL) {}
        TreeNode(int x) : data(x), left(NULL), right(NULL) {}
        TreeNode(int x, TreeNode *left, TreeNode *right) : data(x), left(left), right(right) {}
    };
*/

void preorder(TreeNode* root, vector<int>&v){
    if(root==nullptr) return ;
    v.push_back(root->data);
    preorder(root->left, v);
    preorder(root->right,v);
} 

vector<int> getPreOrderTraversal(TreeNode *root)
{
    vector<int> ans;
    preorder(root, ans);
    return ans;
}

//  post order traversal '
#include <bits/stdc++.h> 
/*
    Following is Binary Tree Node structure:
    class TreeNode
    {
    public:
        int data;
        TreeNode *left, *right;
        TreeNode() : data(0), left(NULL), right(NULL) {}
        TreeNode(int x) : data(x), left(NULL), right(NULL) {}
        TreeNode(int x, TreeNode *left, TreeNode *right) : data(x), left(left), right(right) {}
    };
*/

void postorder(TreeNode* root, vector<int>&v){
    if(root==nullptr) return ;
    v.push_back(root->data);
    postorder(root->right,v);
    postorder(root->left, v);
} 
vector<int> getPostOrderTraversal(TreeNode *root)
{
    vector<int> ans;
    postorder(root, ans);
    reverse(ans.begin() , ans.end());
    return ans;
}

//  left view of binary tree
#include <bits/stdc++.h> 
vector<int> getLeftView(TreeNode<int> *root)
{
    vector<int> ans;
    queue<TreeNode<int>*>q;
    q.push(root);
    if(root==nullptr){
        return ans;
    }
    while(1){
        int size = q.size();
        if(size==0){
            return ans;
        }
        bool flag = true;
        while(size--){
            TreeNode<int>* temp = q.front();
            if(flag) {
                ans.push_back(temp->data);
                flag = !flag;
            }
            q.pop();
            if(temp->left!=NULL){
                q.push(temp->left);
            }
            if(temp->right){
                q.push(temp->right);
            }
        }
    }
    return ans;
}

//  binary tree traversal 
#include <bits/stdc++.h> 
/*************************************************************
 
    Following is the Binary Tree node structure.

    class BinaryTreeNode 
    {
    public : 
        T data;
        BinaryTreeNode<T> *left;
        BinaryTreeNode<T> *right;

        BinaryTreeNode(T data) {
            this -> data = data;
            left = NULL;
            right = NULL;
        }
    };

*************************************************************/

vector<int> bottomView(BinaryTreeNode<int> * root){

    map<int,int > ans;
    queue< pair<BinaryTreeNode<int>* , pair<int,int> > > todo;
    todo.push({root ,  {0,0}});
    while(!todo.empty()){
        auto p = todo.front();
        todo.pop();
        BinaryTreeNode<int> * node = p.first;
        int x = p.second.first;
        int y = p.second.second;
        ans[x] = node->data;
        if(node->left) todo.push({node->left , {x-1 , y+1}});
        if(node->right) todo.push({node->right , {x+1 , y+1}});
    }
    
    vector<int> result;
    for(auto it : ans){
        result.push_back(it.second);
    }
    return result;
}


//  vertical order traversal 
#include <bits/stdc++.h> 

vector<int> verticalOrderTraversal(TreeNode<int> *root)
{
    map<int, vector<pair<int , int>> > nodes ;
    queue<pair<TreeNode<int>* , pair<int,int>>> todo;
    todo.push({root, {0,0}});
    while(!todo.empty()){
        auto p = todo.front();
        todo.pop();
        TreeNode<int>* node = p.first;
        int x = p.second.first;
        int y = p.second.second;
        nodes[x].push_back({node->data , y});
        if(node->left) todo.push({node->left , {x-1 , y+1}});
        if(node->right) todo.push({node->right , {x+1 , y+1}});
    }

    vector<int> ans;
    for(auto it:nodes){
        vector<pair<int,int>>temp = it.second; 
        for(auto m : temp) ans.push_back(m.first);
    }
    return ans;
}

//  Path existing in a tree#include <bits/stdc++.h> 

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

// Maximum width of a tree
#include <bits/stdc++.h> 
/************************************************************

    Following is the TreeNode class structure

    template <typename T>
    class TreeNode {
       public:
        T val;
        TreeNode<T> *left;
        TreeNode<T> *right;

        TreeNode(T val) {
            this->val = val;
            left = NULL;
            right = NULL;
        }
    };

************************************************************/

int getMaxWidth(TreeNode<int> *root)
{
    int width = 0;
    if(!root)return width;
    queue<TreeNode<int>*>q;
    q.push(root);
    while(!q.empty()){
        int sz=q.size();
        width = max(width , sz);
        while(sz--){
            TreeNode<int> *f = q.front();
            q.pop();
            if(f->left)q.push(f->left);
            if(f->right)q.push(f->right);
        }
    }
    return width;
}

// Level order traversal
#include <bits/stdc++.h> 
/************************************************************

    Following is the BinaryTreeNode class structure

    template <typename T>
    class BinaryTreeNode {
       public:
        T val;
        BinaryTreeNode<T> *left;
        BinaryTreeNode<T> *right;

        BinaryTreeNode(T val) {
            this->val = val;
            left = NULL;
            right = NULL;
        }
    };

************************************************************/
vector<int> getLevelOrder(BinaryTreeNode<int> *root)
{
    vector<int> ans;
    queue<BinaryTreeNode<int>*> pq;
    if(!root) return ans;
    pq.push(root);
    while(true){
        int size = pq.size();
        if(size == 0) return ans;
        while(size--){
            BinaryTreeNode<int>* temp = pq.front();
            pq.pop();
            ans.push_back(temp->val);
            if(temp->left) pq.push(temp->left);      
            if(temp->right) pq.push(temp->right);
        }
    }
}

// Height of a tree
#include <bits/stdc++.h> 
int heightOfTheTree(vector<int>& inorder, vector<int>& levelOrder, int n){
	queue<int>q1,q2;
    q1.push(levelOrder[0]);
    int k = 1,height = 0;
    while(!q1.empty() || !q2.empty()){
        if(!q1.empty()) height++;
        while(!q1.empty()){
            int val = q1.front();
			int i = 0;
            for( i = 0;i<n;++i){
                if(inorder[i] == val) break;
            }
            if(i>0 && inorder[i-1] !=-1 && k<n)
                q2.push(levelOrder[k++]);
            if(i<n-1 && inorder[i+1] !=-1 && k<n) 
                q2.push(levelOrder[k++]);
            inorder[i] = -1;
            q1.pop();
        }
        if(!q2.empty()) height++;
        while(!q2.empty()){
            int val = q2.front();
			int i =0;
            for(i = 0;i<n;++i){
                if(inorder[i] == val) break;
            }
            if(i>0 && inorder[i-1] !=-1 && k<n)  
                q1.push(levelOrder[k++]);
            if(i<n-1 && inorder[i+1] !=-1 && k<n) 
                q1.push(levelOrder[k++]);
            inorder[i] = -1;
            q2.pop();
        }
    }
	return height-1;
}

// diameter of a tree
/************************************************************

    Following is the TreeNode class structure

    template <typename T>
    class TreeNode {
       public:
        T data;
        TreeNode<T> *left;
        TreeNode<T> *right;

        TreeNode(T data) {
            this->data = data;
            left = NULL;
            right = NULL;
        }
    };

************************************************************/
int height(TreeNode<int> * node, int &diameter){
    if(!node){
        return 0;
    }
    int lh=height(node->left,diameter);
    int rh=height(node->right,diameter);
    diameter=max(diameter,lh+rh);
    return 1+ max(lh,rh);
}
int diameterOfBinaryTree(TreeNode<int> *root)
{
    int diameter = 0;
    height(root, diameter);
    return diameter;
}


// Is height balance binary tree 
#include <bits/stdc++.h> 
/*************************************************************
 
    Following is the Binary Tree node structure

    class BinaryTreeNode 
    {
    public : 
        T data;
        BinaryTreeNode<T> *left;
        BinaryTreeNode<T> *right;

        BinaryTreeNode(T data) {
            this -> data = data;
            left = NULL;
            right = NULL;
        }
    };

*************************************************************/
int height(BinaryTreeNode<int>* node , bool &result){
    if(!node){
        return 0;
    }
    int lh=height(node->left,result);
    int rh=height(node->right,result);
    result  = result & (abs(lh - rh) <= 1);
    return max(lh,rh) + 1;
}
bool isBalancedBT(BinaryTreeNode<int> *root) {
  bool result = true;
  if (root == nullptr)
    return true;
  height(root, result);
  return result;
}

//  lca of a binary tree 
#include <bits/stdc++.h> 
int lowestCommonAncestor(TreeNode<int> *root, int n1 ,int n2)
{
	if(!root) return -1;
    if(root->data == n1 || root->data == n2) return root->data;
    int lca1 = lowestCommonAncestor(root->left, n1, n2);
    int lca2 = lowestCommonAncestor(root->right, n1, n2);
    if(lca1 != -1 and lca2 != -1) return root->data;
    if(lca1!=-1) return lca1;
    return lca2;
}

//  Check identical trees
#include <bits/stdc++.h> 
bool identicalTrees(BinaryTreeNode<int>* root1, BinaryTreeNode<int>* root2) {
    if(root1 == NULL and root2 == NULL) return true;
    if((root1 != NULL and root2 == NULL ) or (root1 == NULL and root2 != NULL)) return false; 	
    return  (root1->data == root2->data) and identicalTrees(root1->left, root2->left) and identicalTrees(root1->right , root2->right);
}


// binary tree zigzag traversal

#include <bits/stdc++.h> 
/*************************************************************

    Following is the Binary Tree node structure

    class BinaryTreeNode
    {
    public :
        T data;
        BinaryTreeNode<T> *left;
        BinaryTreeNode<T> *right;

        BinaryTreeNode(T data) {
            this -> data = data;
            left = NULL;
            right = NULL;
        }
    };

*************************************************************/

vector<int> zigZagTraversal(BinaryTreeNode<int> *root)
{
    vector<int> ans;
    queue<BinaryTreeNode<int>*> q;
    q.push(root);
    bool flag = 1;
    if (root == nullptr)
        return ans;
    while (1)
    {
        int size = q.size();
        if (size == 0) return ans;
        vector<int> data;
        while (size > 0)
        {
            BinaryTreeNode<int> * temp = q.front();
            data.push_back(temp->data);
            q.pop();
            if (temp->left) q.push(temp->left);
            if (temp->right) q.push(temp->right);
            size--;
        }
        flag = !flag;
        if(flag) reverse(data.begin(), data.end());
        for(auto it: data) ans.push_back(it);
    }
    return ans;
}


// Boundary traversal of a tree

#include <bits/stdc++.h> 
/************************************************************

    Following is the Binary Tree node structure:
    
    template <typename T>
    class TreeNode {
        public :
        T data;
        TreeNode<T> *left;
        TreeNode<T> *right;

        TreeNode(T data) {
            this -> data = data;
            left = NULL;
            right = NULL;
        }

        ~TreeNode() {
            if(left)
                delete left;
            if(right)
                delete right;
        }
    };

************************************************************/

bool isLeafNode(TreeNode<int>* root){
    return (!root->left and !root->right) ;
}

void LeftBoundary(TreeNode<int>* root, vector<int>&ans){
    if(!root) return;
    root = root->left;
    while(root){
    if(!isLeafNode(root)) ans.push_back(root->data);
    if(root->left) root = root->left;
    else root = root->right;
    }
}

void RightBoundary(TreeNode<int>* root, vector<int>&ans){
    if(!root) return;
    root = root->right;
    vector<int> vec;
    while(root){
        if(!isLeafNode(root)) vec.push_back(root->data);
        if(root->right) root = root->right;
        else root = root->left;
    }
    while(vec.size()){
        ans.push_back(vec.back());
        vec.pop_back();
    }
}

void LeafNodes(TreeNode<int>* root , vector<int> &ans){
    if(!root) return;
    if(isLeafNode(root)){
        ans.push_back(root->data); 
    }
    LeafNodes(root->left, ans);
    LeafNodes(root->right, ans);
}

vector<int> traverseBoundary(TreeNode<int>* root){
    vector<int> ans;
    if(!root) return ans;
    if(isLeafNode(root)){
        return {root->data};
    }
    ans.push_back(root->data);
    LeftBoundary(root, ans);
    LeafNodes(root, ans);
    RightBoundary(root, ans);
    return ans;
}

// Maximum pattern between two leaf notes

#include <bits/stdc++.h> 

long long int max(long long int a , long long int b){
    return (a>b) ? a :b;
}

long long int max(long long int a, long long int b , long long int c){
    return (a>max(b,c) ) ? a : (b > max(a,c)) ? b : c;
}
long long int solve(TreeNode<int>* root,long long int &ans){
    if(root == NULL)return 0;
    long long int l = solve(root->left,ans);
    long long int r = solve(root->right,ans);
    ans = max(ans,root->val,root->val+max(l,r,l+r));
    return max(root->val,root->val+max(l,r));
}

bool isLeafNode(TreeNode<int>* root){
    return (root->left == nullptr and root->right == nullptr);
}
long long int findMaxSumPath(TreeNode<int> *root)
{
    if(!root) return -1;
    if(root->left and root->right == nullptr) return -1;
    else if(root->right and root->left == nullptr) return -1;
    if(isLeafNode(root)) return -1;
    long long int ans = 1e-8;
    solve(root , ans);
    return ans;
}

// construct binary tree from preorder and in order
#include <bits/stdc++.h> 



TreeNode<int> * tree(int &index , vector<int> &preorder , vector<int> &inorder, int left , int right ){
    if (left > right) return NULL;
    int pivot = left; 
    while(inorder[pivot] != preorder[index]) pivot++;
    index++;
    TreeNode<int> * newNode = new TreeNode<int>(inorder[pivot]);
    newNode->left = tree(index, preorder, inorder, left, pivot-1);
    newNode->right = tree(index, preorder, inorder, pivot+1, right);
    return newNode;
}
TreeNode<int> *buildBinaryTree(vector<int> &inorder, vector<int> &preorder)
{
    int index = 0;
	return tree(index,preorder, inorder, 0 , inorder.size()-1);
}

// symmetric tree
bool isEqual(BinaryTreeNode<int>* left, BinaryTreeNode<int>*  right) {
    if (!left && !right)
        return true;
    
    if (!left || !right || left->data != right->data)
        return false;
    
    return isEqual(left->left, right->right) && isEqual(left->right, right->left);
}

bool isSymmetric(BinaryTreeNode<int>* root)
{
    if(!root) return true;    // Write your code here.    
    return isEqual(root->left  , root->right);
}


// flatten Binary Tree to a  linked list
#include <bits/stdc++.h> 

void preorder(TreeNode<int>* root , vector<TreeNode<int>*>&a){
    if(!root) return;
    a.push_back(root);
    preorder(root->left,a);
    preorder(root->right,a);
}
TreeNode<int> *flattenBinaryTree(TreeNode<int> *root)
{
    TreeNode<int>* result;
    if(root==NULL) return NULL;
    if(root->left==NULL && root->right==NULL) return  root;
    else{
        result = root;
        vector<TreeNode<int>*>ans;
        preorder(root,ans);
        reverse(ans.begin(),ans.end());
        root = NULL;
        root =  ans.back();
        ans.pop_back();
        while(ans.size()){
            root->left = NULL;
            root->right = ans.back();
            root = root->right;
            ans.pop_back();
        }
    }
    return result;
}

// invert a binary tree

#include <bits/stdc++.h> 

bool extract_path(TreeNode<int> *root, TreeNode<int> *leaf, stack<TreeNode<int>*> &path) {
    if(root == NULL) return 0;

    path.push(root);

    if(root->data == leaf->data) return 1;
    bool lf = extract_path(root->left, leaf, path);
    bool rg = extract_path(root->right, leaf, path);

    if(lf || rg) return 1;

    path.pop();
    return 0;
}

TreeNode<int> * invertBinaryTree(TreeNode<int> *root, TreeNode<int> *leaf)
{
	// Write your code here.
    if(root == NULL or leaf == NULL) return NULL;
    
    stack<TreeNode<int>*> path;
    extract_path(root, leaf, path);


    TreeNode<int> *parent = path.top();
    path.pop();

    TreeNode<int> *new_root = parent;

    while(!path.empty()) {
        TreeNode<int>* cur = path.top();
        path.pop();

        if(cur->left == parent) {
            cur->left = NULL;
        }

        else {
            cur->right = cur->left;
            cur->left = NULL;
        }

        parent->left = cur;

        parent = cur;
    }

    return new_root;
}


// children sum property
#include <bits/stdc++.h> 

void changeTree(BinaryTreeNode < int > * root) {
    // Write your code here.

    if(!root) return;

    int child = 0;
    if(root->left) child += root->left->data;   
    if(root->right) child += root->right->data;

    if(root->data <= child) root->data = child;
    else {
        if(root->left) root->left->data = root->data;
        else if(root->right) root->right->data = root->data;
    }
    changeTree(root->left);
    changeTree(root->right);
    
    int value = 0;
    if(root->left) value += root->left->data;
    if(root->right) value += root->right->data;
    if(root->left or root->right) root->data = value;

}  

//  Count nodes

#include <bits/stdc++.h> 

void connectNodes(BinaryTreeNode< int > *root) {
    if(!root) return ;
    queue<BinaryTreeNode< int > *> q;
    q.push(root);        
    while(q.size()) {
        BinaryTreeNode< int > * rightNode = nullptr;                    
        int size = q.size();
        while(size--) {                
            auto cur = q.front(); q.pop();           
            cur -> next = rightNode;                 
            rightNode = cur;                          
            if(cur -> right) q.push(cur -> right);
            if(cur ->left ) q.push(cur -> left);                  
        }
    }
}

// search in a bst

#include <bits/stdc++.h> 
/*
    Following is the Binary Tree node structure:

    class BinaryTreeNode {
    public : 
        T data;
        BinaryTreeNode<T> *left;
        BinaryTreeNode<T> *right;

        BinaryTreeNode(T data) {
            this -> data = data;
            left = NULL;
            right = NULL;
        }
    };
*/

bool searchInBST(BinaryTreeNode<int> *root, int x) {
    // Write your code here.
    if(!root) return false;
    if(root->data == x) return true;
    else if(root->data < x){
        return searchInBST(root->right, x);
    }
    else{
        return searchInBST(root->left ,x);
    }
    return false;
}

// Convert sorted array to B S T

#include <bits/stdc++.h> 
/************************************************************

    Following is the TreeNode class structure

    template <typename T>
    class TreeNode {
       public:
        T val;
        TreeNode<T> *left;
        TreeNode<T> *right;
        
        TreeNode(T val) {
            this->val = val;
            left = NULL;
            right = NULL;
        }
    };

************************************************************/
TreeNode<int>* sortedTree(vector<int> &nums , int start , int end){
    if(start==end) return NULL;
    int mid = (start+end)/2;
    TreeNode<int>* res = new TreeNode<int>(nums[mid]);
    res->left = sortedTree(nums, start, mid);
    res->right = sortedTree(nums, mid+1, end);
    return res;
}
TreeNode<int>* sortedArrToBST(vector<int> &arr, int n)
{
    return sortedTree(arr,0,n);
}

//  construct binary search tree from preorder and inorder

#include <bits/stdc++.h> 

TreeNode<int>* insertIntoBST(TreeNode<int>* root, int data) {
    TreeNode<int>* parent = NULL;
    TreeNode<int>* curr = root ;
    while(curr!=NULL){
        parent = curr;
        if(curr->data > data) curr = curr->left;
        else if(curr->data < data) curr = curr->right;
        else return root;
    }
    if(parent==NULL) return new TreeNode<int>(data);
    else if(parent->data > data) parent->left = new TreeNode<int>(data);
    else parent->right = new TreeNode<int>(data);
    return root;
}

TreeNode<int>* preOrderTree(vector<int> &preOrder){
    TreeNode<int> *r = nullptr;
    for(auto it : preOrder){
        r = insertIntoBST(r, it);
    }
    return r;
}

// lca of two numbers in a bst 
#include <bits/stdc++.h> 


TreeNode<int>* LCAinaBST(TreeNode<int>* root, TreeNode<int>* p, TreeNode<int>* q)
{
    if(!root) return NULL;
    if(root->data == p->data or root->data == q->data) return root;
    if(root->data > p->data and root->data < q->data) return root;
    else if(root->data < p->data and root->data > q->data) return root;
    if(root->data > p->data and root->data > q->data) return LCAinaBST(root->left,p,q);
    else if(root->data < p->data and root->data < q->data) return LCAinaBST(root->right, p, q);
}


//  predeccessor and successor of a bst 
#include <bits/stdc++.h> 
/*************************************************************
*************************************************************/



void inorder(BinaryTreeNode<int>* root, vector<int>&ans){
    if(!root) return;
    inorder(root->left, ans);
    ans.push_back(root->data);
    inorder(root->right,ans);
}
pair<int,int> predecessorSuccessor(BinaryTreeNode<int>* root, int key)
{
    vector<int> ans = {-1};
    inorder(root, ans);
    ans.push_back(-1);
    pair<int, int> pr;
    for(int i=1; i<ans.size()-1;i++){
        if(ans[i] == key){
            pr = {ans[i-1] , ans[i+1]};
            return pr;
        }
    }
}

//  floor in a bst 
#include <bits/stdc++.h> 

void floor(TreeNode<int>* root , int x, int &result){
    if(!root) return;
    if(root->val == x){
        result = root->val;
        return;
    }
    else if(root->val < x){
        result = max(result , root->val);
        floor(root->right, x, result);
    }
    else if(root->val > x){
        floor(root->left , x, result);
    }
}

int floorInBST(TreeNode<int> * root, int x)
{
    int result = INT_MIN;
    floor(root , x, result);
    return result;
}

// ceil from a bst 
#include <bits/stdc++.h> 
void inorder(BinaryTreeNode<int> * root, int key , int &result){
    if(root==nullptr) return ;
    inorder(root->left, key , result);
    if(root->data >= key and result > root->data) result = root->data;
    inorder(root->right,key , result);
}
int findCeil(BinaryTreeNode<int> *node, int x){
    // Write your code here.
    int rs = INT_MAX;
    inorder(node, x, rs);
    return rs == INT_MAX ? -1 : rs;
}

// /kth largest number bst 

#include <bits/stdc++.h> 
/************************************************************
    Following is the Binary Search Tree node structure
    
    template <typename T>
    class TreeNode {
        public :
        T data;
        TreeNode<T> *left;
        TreeNode<T> *right;

        TreeNode(T data) {
            this -> data = data;
            left = NULL;
            right = NULL;
        }

        ~TreeNode() {
            if(left)
                delete left;
            if(right)
                delete right;
        }
    };

************************************************************/
void inorder(TreeNode<int>* root, vector<int>&v){
    if(root==nullptr) return ;
    inorder(root->left, v);
    v.push_back(root->data);
    inorder(root->right,v);
} 

    
int KthLargestNumber(TreeNode<int>* root, int k) 
{
   vector<int> ans;
   inorder(root, ans);
   return (ans.size() >= k) ? ans[ans.size()-k] : -1;
}



//  01 knapsack 

int maxProfit(vector<int> &values, vector<int> &weights, int n, int W)
{
	vector<vector<int>> d(n+1, vector<int> (W+1,0));
	for (int i = 1; i <=n ; i++)
	{
		for (int w = 1; w <= W; w++)
		{                    
			if (weights[i-1] <= w)
			{
				d[i][w] = max(d[i-1][w],( d[i-1][ w - weights[i-1]] + values[i-1]));
			}
			else
			{
				d[i][w] = d[i-1][w];
			}                    
		}
	}
	return d[n][W];
}






