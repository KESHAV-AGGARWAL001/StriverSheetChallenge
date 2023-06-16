#include <bits/stdc++.h>
long long pow(int a, int b, int MOD)
{
    if (b == 0)
        return 1;
    long long res = pow(a, b / 2, MOD);
    res *= res;
    res %= MOD;
    if (b % 2)
        res *= a;
    res %= MOD;
    return res;
}
int modularExponentiation(int x, int n, int m)
{
    long long odd = n / 2;
    long long even = n / 2 + n % 2;
    return (pow(x, even, m) * pow(x, odd, m)) % m;
}