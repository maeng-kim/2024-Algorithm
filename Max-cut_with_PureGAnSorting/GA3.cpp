#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <set>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <ctime>

using namespace std;

struct Individual {
    vector<int> chromosome;
    int fitness;

    Individual() : fitness(0) {}
    
    Individual(int size) : chromosome(size), fitness(0) {}

    Individual(const Individual& other) : chromosome(other.chromosome), fitness(other.fitness) {}

    Individual& operator=(const Individual& other) {
        if (this != &other) {
            chromosome = other.chromosome;
            fitness = other.fitness;
        }
        return *this;
    }

    Individual(Individual&& other) noexcept : chromosome(std::move(other.chromosome)), fitness(other.fitness) {}

    Individual& operator=(Individual&& other) noexcept {
        if (this != &other) {
            chromosome = std::move(other.chromosome);
            fitness = other.fitness;
        }
        return *this;
    }
};

class GeneticAlgorithm {
public:
    vector<vector<pair<int, int>>> adj;
    vector<Individual> population;
    int V, E;
    double baseMutationRate;
    Individual bestIndividual;
    int populationSize;
    int generations;
    mt19937 gen;
    double bestFitness = numeric_limits<double>::lowest();
    double averageFitness = 0;
    
    GeneticAlgorithm(double mutationRate, int popSize, int gens)
        : V(0), E(0), baseMutationRate(mutationRate), populationSize(popSize) {
        random_device rd;
        gen = mt19937(rd());
    }
    
    void loadGraph(const string& filename) {
        ifstream file(filename);
        if (!file) {
            cerr << "Unable to open file" << endl;
            return;
        }
        file >> V >> E;
        adj.resize(V);
        int u, v, weight;
        while (file >> u >> v >> weight) {
            adj[u].push_back(make_pair(v, weight));
            adj[v].push_back(make_pair(u, weight));
        }
        file.close();
    }
    
    void initializePopulation() {
        uniform_int_distribution<> dis(0, 1);
        population.clear();
        for (int i = 0; i < populationSize; ++i) {
            Individual ind(V);
            for (int& gene : ind.chromosome) {
                gene = dis(gen);
            }
            population.push_back(ind);
        }
    }
    
    void calculateFitness() {
        for (auto& ind : population) {
            ind.fitness = 0;
            for (int u = 0; u < V; ++u) {
                for (const auto& edge : adj[u]) {
                    int v = edge.first;
                    if (ind.chromosome[u] != ind.chromosome[v])
                        ind.fitness += edge.second;
                }
            }
        }
    }
    
    void mutate(Individual& ind) {
        uniform_real_distribution<> dis(0.0, 1.0);
        double mutationRate = (ind.fitness < 0) ? baseMutationRate * 2 : baseMutationRate;
        for (int& gene : ind.chromosome) {
            if (dis(gen) < mutationRate) {
                gene ^= 1; // Toggle the gene
            }
        }
    }
    
    void uniformCrossover(const Individual& parent1, const Individual& parent2, Individual& child) {
        uniform_int_distribution<> dis(0, 1);
        for (int i = 0; i < V; ++i) {
            child.chromosome[i] = dis(gen) ? parent1.chromosome[i] : parent2.chromosome[i];
        }
    }

    
    Individual rouletteWheelSelection() {
        double totalFitness = accumulate(population.begin(), population.end(), 0.0,
                                         [](double sum, const Individual& ind) {
                                            return sum + exp(ind.fitness);
                                         });
        uniform_real_distribution<> dis(0.0, totalFitness);
        double value = dis(gen);

        double partialSum = 0.0;
        for (auto& ind : population) {
            partialSum += exp(ind.fitness);
            if (partialSum >= value) {
                return ind;
            }
        }
        return population.back();
    }
    
    Individual tournamentSelection(int tournamentSize) {
        vector<int> indices(populationSize);
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), gen);

        int bestIndex = indices[0];
        for (int i = 1; i < tournamentSize; ++i) {
            if (population[indices[i]].fitness > population[bestIndex].fitness) {
                bestIndex = indices[i];
            }
        }
        return population[bestIndex];
    }

    
    /*
     
     1. merge sort
     2. basic quick sort
     3. intelligent quick sort
     4. paranoid quick sort
     5. counting sort
     */
    
    
    /*
     1. MergeSort
     */
   /* void merge(vector<Individual>& arr, size_t l, size_t m, size_t r) {
        size_t n1 = m - l + 1;
        size_t n2 = r - m;

        vector<Individual> L(arr.begin() + l, arr.begin() + m + 1);
        vector<Individual> R(arr.begin() + m + 1, arr.begin() + r + 1);

        size_t i = 0, j = 0, k = l;

        
        while (i < n1 && j < n2) {
            if (L[i].fitness >= R[j].fitness) {
                arr[k] = L[i++];
            } else {
                arr[k] = R[j++];
            }
            k++;
        }

        while (i < n1) {
            arr[k++] = L[i++];
        }

        while (j < n2) {
            arr[k++] = R[j++];
        }
    }

    void mergeSort(vector<Individual>& arr, size_t l, size_t r) {
        if (l >= r)
            return;

        size_t m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }*/

    
    /*
     2. basic quick sort
     */
   /* int partition(vector<Individual>& arr, int low, int high) {
        Individual pivot = arr[high];
        int i = low - 1;

        for (int j = low; j < high; j++) {
            if (arr[j].fitness >= pivot.fitness) {
                swap(arr[++i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[high]);
        return i + 1;
    }
    void quickSort(vector<Individual>& arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }*/
    
    
    /*
     3. intelligent quick sort
     */
    Individual medianOfThree(const Individual& a, const Individual& b, const Individual& c) {
        if ((a.fitness >= b.fitness && a.fitness <= c.fitness) || (a.fitness >= c.fitness && a.fitness <= b.fitness)) {
            return a;
        } else if ((b.fitness >= a.fitness && b.fitness <= c.fitness) || (b.fitness >= c.fitness && b.fitness <= a.fitness)) {
            return b;
        } else {
            return c;
        }
    }

    int partitionIntelligent(vector<Individual>& arr, int low, int high) {
        int mid = low + (high - low) / 2;
        Individual pivot = medianOfThree(arr[low], arr[mid], arr[high]);
        while (low <= high) {
            while (arr[low].fitness > pivot.fitness) low++;
            while (arr[high].fitness < pivot.fitness) high--;
            if (low <= high) {
                swap(arr[low], arr[high]);
                low++;
                high--;
            }
        }
        return low;
    }


    void intelligentQuickSort(vector<Individual>& arr, int low, int high) {
        if (low < high) {
            int pi = partitionIntelligent(arr, low, high);
            intelligentQuickSort(arr, low, pi - 1);
            intelligentQuickSort(arr, pi, high);
        }
    }
    
    
    /*
     4. paranoid quick sort
     */
   /* int findMaxIndex(vector<Individual>& arr, int low, int high) {
        int maxIndex = low;
        for (int i = low + 1; i <= high; i++) {
            if (arr[i].fitness > arr[maxIndex].fitness) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    int partitionParanoid(vector<Individual>& arr, int low, int high) {
        int maxIndex = findMaxIndex(arr, low, high);
        swap(arr[maxIndex], arr[high]);
        int pivot = arr[high].fitness;
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j].fitness > pivot) {
                i++;
                swap(arr[i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[high]);
        return i + 1;
    }

    void paranoidQuickSort(vector<Individual>& arr, int low, int high) {
        if (low < high) {
            int pi = partitionParanoid(arr, low, high);
            paranoidQuickSort(arr, low, pi - 1);
            paranoidQuickSort(arr, pi + 1, high);
        }
    }*/

    
    /*
     5. counting sort
     */
   /* void countingSort(vector<Individual>& arr) {
        if (arr.empty()) return;

        int maxFitness = arr[0].fitness;
        int minFitness = arr[0].fitness;
        for (const auto& ind : arr) {
            if (ind.fitness > maxFitness) maxFitness = ind.fitness;
            if (ind.fitness < minFitness) minFitness = ind.fitness;
        }

        //cout << "Max fitness: " << maxFitness << ", Min fitness: " << minFitness << endl;

        size_t range = static_cast<size_t>(maxFitness - minFitness + 1);
        vector<size_t> count(range, 0);
        
        for (const auto& individual : arr) {
            count[individual.fitness - minFitness]++;
        }

        for (size_t i = 1; i < range; ++i) {
            count[i] += count[i - 1];
        }

        vector<Individual> output(arr.size());
        for (int i = arr.size() - 1; i >= 0; --i) {
            output[count[arr[i].fitness - minFitness] - 1] = arr[i];
            count[arr[i].fitness - minFitness]--;
        }

        arr = std::move(output);

    }*/
    
    void updateBestAndAverage() {
        double totalFitness = 0;
        for (auto& ind : population) {
            totalFitness += ind.fitness;
            if (ind.fitness > bestFitness) {
                bestFitness = ind.fitness;
                bestIndividual = ind;
            }
        }
        averageFitness = totalFitness / populationSize;
    }

    
    void run() {
        ofstream outFile("maxcut.out");
        initializePopulation();
        calculateFitness();
        vector<Individual> newPopulation;
        
        auto start = chrono::steady_clock::now();
        auto timeout = chrono::seconds(180);
       // int generations = 0;

        while (chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - start) < timeout) {
              vector<Individual> newPopulation;
              newPopulation.reserve(populationSize);
              intelligentQuickSort(population, 0, population.size()-1);
              int elitismCnt = populationSize /2;
              newPopulation.insert(newPopulation.end(), population.begin(), population.begin() + elitismCnt);
              // Fill the rest of the population with offspring
              while (newPopulation.size() < populationSize) {
                  Individual parent1 = tournamentSelection(5);
                  Individual parent2 = tournamentSelection(5);
                  Individual child(V);
                  uniformCrossover(parent1, parent2, child);
                  mutate(child);
                  newPopulation.push_back(child);
              }

              population =std:: move(newPopulation);
              calculateFitness();
              generations++;

              // Update the best and average fitness
              updateBestAndAverage();
          }

          //cout << "Generations run: " << generations << endl;
          //cout << "Best Fitness: " << bestFitness << endl;
          //cout << "Average Fitness: " << averageFitness << endl;
          //cout << "Vertices in the Best Solution: ";
          for (int i = 0; i < bestIndividual.chromosome.size(); ++i) {
              if (bestIndividual.chromosome[i] == 1) {
                  outFile << i << " ";
              }
          }
        outFile.close();
    }
    
};

int main() {
    GeneticAlgorithm ga(0.01, 500, 1);
    ga.loadGraph("maxcut.in");
    ga.run();
    return 0;
}




