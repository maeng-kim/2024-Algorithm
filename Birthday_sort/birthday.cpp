#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
using namespace std;

static int partition (vector<pair<string, string> >& a, int left, int right);

void quickSort(vector<pair<string, string> >& a, int left, int right) {
    if(left < right) {
        int q = partition(a, left, right);
        quickSort(a, left, q - 1);
        quickSort(a, q + 1, right);
    }
}

int partition(vector<pair<string, string> >& a, int left, int right) {
    int low = left + 1;
    int high = right;
    int pivot = stoi(a[left].second.substr(0,2));

    while (low <= high) {
        while (low <= right && stoi(a[low].second.substr(0,2)) < pivot) low++;
        while (high >= left && stoi(a[high].second.substr(0,2)) > pivot) high--;
        
        if (low <= high) {
            swap(a[low], a[high]);
            low++;
            high--;
        }
    }

    swap(a[left], a[high]);
    return high;
}

int main() {
    vector<pair<string, string> > list;
    ifstream input("birthday.in");
    if(!input.is_open()) {
        cout << "can't open the file\n";
        return 1;
    }

    string line, name, birth;
    while(getline(input, line)) {
        stringstream ss(line);
        ss >> name >> birth;
        list.push_back(make_pair(name, birth));
    }
    input.close();

    quickSort(list, 0, list.size() - 1);
    ofstream file;
    file.open("birthday.out");
    cout.rdbuf(file.rdbuf());
    
    for(const auto& pair : list) {
        cout << pair.first << " " << pair.second << endl;
    }

    return 0;
}

