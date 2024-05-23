#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <set>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <ctime>
#include <limits>

using namespace std;

struct Individual {
    vector<int> chromosome;
    int fitness;

    Individual() : fitness(0) {}
    Individual(int size) : chromosome(size), fitness(0) {}
};

class Island {
public:
    vector<Individual> population;
    vector<vector<pair<int, int>>> adj;
    int V, E;
    mt19937 gen;
    int populationSize;
    double mutationRate;

    Island(int popSize, double mutRate, const vector<vector<pair<int, int>>>& adjacency) : adj(adjacency), populationSize(popSize), mutationRate(mutRate) {
        random_device rd;
        gen = mt19937(rd());
        V = adj.size();
        initializePopulation();
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
        calculateFitness();
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

    void runGeneration() {
        vector<Individual> newPopulation;
        for (int i = 0; i < populationSize; ++i) {
            Individual parent1 = tournamentSelection(5);
            Individual parent2 = tournamentSelection(5);
            Individual child(V);
            uniformCrossover(parent1, parent2, child);
            mutate(child);
            newPopulation.push_back(child);
        }
        population = std::move(newPopulation);
        calculateFitness();
    }

    void runLocalOptimization(int maxIterations) {
        bool improved = true;
        int iteration = 0;
        while (improved && iteration < maxIterations) {
            improved = false;
            for (auto& ind : population) {
                for (int i = 0; i < V; ++i) {
                    int delta = calculateDelta(ind, i);
                    if (delta > 0) {
                        ind.chromosome[i] = 1 - ind.chromosome[i];
                        ind.fitness += delta;
                        improved = true;
                    }
                }
            }
            iteration++;
        }
    }

    int calculateDelta(Individual& ind, int index) {
        int delta = 0;
        for (auto& edge : adj[index]) {
            if (ind.chromosome[index] == ind.chromosome[edge.first])
                delta -= edge.second;
            else
                delta += edge.second;
        }
        return delta;
    }

    const Individual& getBestIndividual() const {
        return *max_element(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
            return a.fitness < b.fitness;
        });
    }

    double getAverageFitness() const {
        double totalFitness = accumulate(population.begin(), population.end(), 0.0,
                                         [](double sum, const Individual& ind) {
                                             return sum + ind.fitness;
                                         });
        return totalFitness / population.size();
    }

    void sortPopulation() {
        sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
            return a.fitness > b.fitness;
        });
    }
};

class GeneticAlgorithm {
public:
    vector<vector<pair<int, int>>> globalAdj;
    vector<Island> islands;
    int numIslands;
    int globalV, globalE;
    int generations;
    int migrationInterval;
    mt19937 gen;

    GeneticAlgorithm(int numIslands, int popSize, int gens, double mutationRate, int migrateInterval, const string& filename) : numIslands(numIslands), generations(gens), migrationInterval(migrateInterval) {
        random_device rd;
        gen = mt19937(rd());
        loadGraph(filename);
        for (int i = 0; i < numIslands; i++) {
            islands.emplace_back(popSize, mutationRate, globalAdj);
        }
    }

    void loadGraph(const string& filename) {
        ifstream file(filename);
        if (!file) {
            throw runtime_error("Unable to open file: " + filename);
        }
        file >> globalV >> globalE;
        if (file.fail() || globalV <= 0 || globalE <= 0) {
            throw runtime_error("Error reading graph size from file.");
        }
        globalAdj.resize(globalV);
        int u, v, weight;
        while (file >> u >> v >> weight) {
            if (u >= globalV || v >= globalV || u < 0 || v < 0) {
                //cerr << "Ignoring invalid edge (" << u << ", " << v << ") with weight " << weight << endl;
                continue; // 무시
            }
            globalAdj[u].push_back(make_pair(v, weight));
            globalAdj[v].push_back(make_pair(u, weight));
        }
        file.close();
    }

    void run() {
        auto startTime = chrono::high_resolution_clock::now();
        int generation = 0;

        while (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - startTime).count() < 120) {
            for (auto& island : islands) {
                island.runGeneration();
            }

            if (generation % migrationInterval == 0) {
                migrate();
            }

            if (generation % 100 == 0) {
                printFitnessStatistics(generation);
            }

            generation++;

            if (generation >= generations) break;
        }

        mergeIslandsAndOptimize(startTime);
        printFinalResults();
    }

    void migrate() {
        for (int i = 0; i < numIslands; ++i) {
            int next = (i + 1) % numIslands;
            const Individual& bestInd = islands[i].getBestIndividual();
            islands[next].population[0] = bestInd;
        }
    }

    void mergeIslandsAndOptimize(const chrono::high_resolution_clock::time_point& startTime) {
        vector<Individual> mergedPopulation;
        for (auto& island : islands) {
            mergedPopulation.insert(mergedPopulation.end(), island.population.begin(), island.population.end());
        }

        Island globalIsland(mergedPopulation.size(), 0.01, globalAdj);
        globalIsland.population = mergedPopulation;
        globalIsland.calculateFitness();

        while (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - startTime).count() < 180) {
            globalIsland.runGeneration();
            if (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - startTime).count() >= 180) break;
        }

        globalIsland.runLocalOptimization(100);

        // 최적 해의 정점 번호 출력
        ofstream outFile("maxcut.out");
        const Individual& best = globalIsland.getBestIndividual();

        for (size_t i = 0; i < best.chromosome.size(); ++i) {
            if (best.chromosome[i] == 1) {
                outFile << i << " ";
            }
        }
        outFile << endl;
        outFile.close();
        
        //double averageFitness = globalIsland.getAverageFitness();
        //cout << "Average Fitness: " << averageFitness << endl;
    }

    void printFitnessStatistics(int generation) {
        for (int i = 0; i < numIslands; ++i) {
            //const Individual& best = islands[i].getBestIndividual();
            //double avg = islands[i].getAverageFitness();
            //cout << "Generation " << generation << " - Island " << i << " Best Fitness: " << best.fitness << ", Average Fitness: " << avg << endl;
        }
    }

    void printFinalResults() {
        // 이미 mergeIslandsAndOptimize에서 최적 해를 출력했으므로 여기서는 생략합니다.
    }
};

int main() {
    try {
        GeneticAlgorithm ga(4, 100, 1000, 0.01, 10, "maxcut.in");
        ga.run();
    } catch (const exception& e) {
        cerr << "Exception: " << e.what() << endl;
        return 1;
    }
    return 0;
}
