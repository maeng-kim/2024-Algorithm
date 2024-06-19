#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <unordered_map>
#include <tuple>
#include <functional>

using namespace std;

// 카드 덱 생성 및 셔플
vector<int> createDeck() {
    vector<int> deck;
    for (int i = 1; i <= 13; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i > 10) {
                deck.push_back(10); // J, Q, K는 10으로 계산
            } else {
                deck.push_back(i);
            }
        }
    }
    random_device rd;
    mt19937 g(rd());
    shuffle(deck.begin(), deck.end(), g);
    return deck;
}

// 카드 합 계산
int calculateScore(const vector<int> &hand) {
    int score = 0;
    int aceCount = 0;
    for (int card : hand) {
        if (card == 1) {
            aceCount++;
            score += 11; // Ace는 11로 계산
        } else {
            score += card;
        }
    }
    while (score > 21 && aceCount > 0) {
        score -= 10; // Ace를 1로 계산
        aceCount--;
    }
    return score;
}

// 커스텀 해시 함수 정의
struct TupleHash {
    template <class T1, class T2, class T3, class T4>
    size_t operator()(const tuple<T1, T2, T3, T4>& t) const {
        auto hash1 = hash<T1>{}(get<0>(t));
        auto hash2 = hash<T2>{}(get<1>(t));
        auto hash3 = hash<T3>{}(get<2>(t));
        auto hash4 = hash<T4>{}(get<3>(t));
        return hash1 ^ (hash2 << 1) ^ (hash3 << 2) ^ (hash4 << 3); // 간단한 해시 조합 방법
    }
};

// 동적 프로그래밍을 위한 메모이제이션 테이블
unordered_map<tuple<int, int, int, bool>, int, TupleHash> memo;

int optimal_strategy(int player_score, int dealer_score, int card_index, const vector<int> &deck, bool is_player_turn) {
    if (player_score > 21) return -1; // 플레이어가 버스트(21점을 넘음)
    if (dealer_score > 21) return 1;  // 딜러가 버스트(21점을 넘음)

    // 모든 카드를 사용한 경우, 현재 점수 비교
    if (card_index >= deck.size()) {
        if (player_score > dealer_score) return 1; // 플레이어 승리
        if (player_score < dealer_score) return -1; // 딜러 승리
        return 0; // 무승부
    }

    auto state = make_tuple(player_score, dealer_score, card_index, is_player_turn);
    if (memo.find(state) != memo.end()) return memo[state]; // 이미 계산된 상태 반환

    int next_card = deck[card_index];
    int result;

    if (is_player_turn) {
        // 플레이어의 턴: 히트와 스탠드 중 최적 선택
        int hit = optimal_strategy(player_score + next_card, dealer_score, card_index + 1, deck, true);
        int stand = optimal_strategy(player_score, dealer_score, card_index + 1, deck, false);
        result = max(hit, stand);
    } else {
        // 딜러의 턴: 17 이상이면 스탠드, 아니면 히트
        if (dealer_score >= 17) {
            result = optimal_strategy(player_score, dealer_score, card_index + 1, deck, true);
        } else {
            result = optimal_strategy(player_score, dealer_score + next_card, card_index + 1, deck, false);
        }
    }

    memo[state] = result;
    return result;
}

int playGameWithDP() {
    vector<int> deck = createDeck();
    vector<int> playerHand, dealerHand;

    // 초기 카드 분배
    playerHand.push_back(deck.back()); deck.pop_back();
    dealerHand.push_back(deck.back()); deck.pop_back();
    playerHand.push_back(deck.back()); deck.pop_back();
    dealerHand.push_back(deck.back()); deck.pop_back();

    int playerScore = calculateScore(playerHand);
    int dealerScore = calculateScore(dealerHand);

    // 최적의 전략을 찾기 위해 동적 프로그래밍 사용
    int result = optimal_strategy(playerScore, dealerScore, 0, deck, true);
    return result;
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    int totalGames = 1000;
    int playerWins = 0;
    int dealerWins = 0;
    int ties = 0;

    for (int i = 0; i < totalGames; ++i) {
        memo.clear(); // 각 게임마다 메모이제이션 테이블 초기화
        int result = playGameWithDP();
        if (result == 1) {
            playerWins++;
        } else if (result == -1) {
            dealerWins++;
        } else {
            ties++;
        }
    }

    cout << "Total games: " << totalGames << endl;
    cout << "Player wins: " << playerWins << " (" << (playerWins / (double)totalGames * 100) << "%)" << endl;
    cout << "Dealer wins: " << dealerWins << " (" << (dealerWins / (double)totalGames * 100) << "%)" << endl;
    cout << "Ties: " << ties << " (" << (ties / (double)totalGames * 100) << "%)" << endl;

    return 0;
}

