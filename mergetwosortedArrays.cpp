#include <bits/stdc++.h>

// vector<int> ninjaAndSortedArrays(vector<int> &nums1, vector<int> &nums2, int m, int n)
// {
//     int len = nums1.size() - m;
//     while (len--)
//     {
//         nums1.pop_back();
//     }
//     for (int i = 0; i < n; i++)
//     {
//         nums1.push_back(nums2[i]);
//     }
//     sort(nums1.begin(), nums1.end());
//     return nums1;
// }

vector<int> mergeSortedArrays(vector<int> &nums1, vector<int> &num2, int m, int n)
{
    int len = nums1.size() - m;
    while (len--)
    {
        nums1.pop_back();
    }

    int i = 0, j = 0;
    vector<int> ans;
    while (i < m and j < n)
    {
        if (nums1[i] < nums[j])
        {
            ans.push_back(nums1[i++]);
        }
        else
        {
            ans.push_back(nums2[j++]);
        }
    }
    return ans;
}