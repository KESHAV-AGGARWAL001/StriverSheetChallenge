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













