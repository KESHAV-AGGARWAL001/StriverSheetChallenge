bool searchMatrix(vector<vector<int>> &matrix, int target)
{
    // for (int i = 0; i < matrix.size(); i++)
    // {
    //     for (int j = 0; j < matrix[0].size(); j++)
    //     {
    //         if (matrix[i][j] == target)
    //             return true;
    //     }
    // }
    // return false;

    int row = matrix.size(), col = matrix[0].size();
    int start = 0;
    int end = row * col - 1;

    while (start <= end)
    {
        int mid = start + (end - start) / 2;
        int ele = matrix[mid / col][mid % col];
        if (ele == target)
        {
            return 1;
        }
        if (ele < target)
        {
            start = mid + 1;
        }
        else
        {
            end = mid - 1;
        }
    }
    return 0;
}