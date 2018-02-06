#include "mult.h"
#include <iostream>
using namespace std;

int main() {
    {
        Mult mult({3, 2, 1, 3});
        for (auto b: mult.bit)
            cout << b << ' ';
        cout << endl;

        cout << mult.Query(0) << endl; // 0
        cout << mult.Query(2) << endl; // 0
        cout << mult.Query(4) << endl; // 1
        cout << mult.Query(5.5) << endl; // 2
        cout << mult.Query(7) << endl; // 3
        cout << mult.Query(10) << endl; // 4
    }

    {
       Mult mult({3, 2, 1, 3, 4});
       cout << mult.Query(0) << endl; // 0
       cout << mult.Query(2) << endl; // 0
       cout << mult.Query(4) << endl; // 1
       cout << mult.Query(5.5) << endl; // 2
       cout << mult.Query(7) << endl; // 3
       cout << mult.Query(10) << endl; // 4
       cout << mult.Query(14) << endl; // 5
    }

    {
        Mult mult({1, 0.1, 100, 10000, 1000});
        //for (auto b: mult.bit)
        //    cout << b << ' ';
        //cout << endl;
        // 3 4 2 0 1
        for (int i = 0; i < 5; i++) {
            cout << mult.Query() << endl;
            //for (auto b: mult.bit)
            //    cout << b << ' ';
            //cout << endl;
        }
    }
}
