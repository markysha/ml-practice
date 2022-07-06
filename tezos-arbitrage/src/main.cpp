#include <fstream>
#include <vector>
#include <optional>
#include <iostream>
#include <sstream>
#include <queue>
#include <math.h>
#include <set>
#include <string>
#include <array>

const double EPS = 1e-3;

struct CEXTrade {
    uint64_t id;
    long double price;
    long double qty;
    long double quote_qty;
    uint64_t time;
    bool is_buyer_maker;
};

std::ifstream &operator>>(std::ifstream &ifs, CEXTrade &trade) {
    char d;
    std::string s;
    ifs >> trade.id
        >> d
        >> trade.price
        >> d
        >> trade.qty
        >> d
        >> trade.quote_qty
        >> d
        >> trade.time
        >> d
        >> s;
    trade.is_buyer_maker = s == "True";
    ifs.get();
    return ifs;
}

struct DEXTrade {
    uint64_t id;
    std::string hash;
    uint64_t time;
    uint32_t op_n;
    uint32_t op_p;
    std::string status;
    int gas_limit;
    int gas_used;
    long double volume;
    long double fee;
    std::string sender;
    std::string block;
    std::string entrypoint;
    long double token_pool;
    long double tez_pool;
};

std::ifstream &operator>>(std::ifstream &ifs, DEXTrade &trade) {
    int x;
    ifs >> x
        >> trade.id
        >> trade.hash
        >> trade.time
        >> trade.op_n
        >> trade.op_p
        >> trade.status
        >> trade.gas_limit
        >> trade.gas_used
        >> trade.volume
        >> trade.fee
        >> trade.sender
        >> trade.block;
    ifs.get();
    if (ifs.peek() != ' ') {
        ifs >> trade.entrypoint;
    }
    ifs >> trade.token_pool
        >> trade.tez_pool;
//    ifs.get();
    return ifs;
}

struct Exchange {
    long double token_pool;
    long double tez_pool;
    long double fee = 0;

    Exchange() : token_pool(0), tez_pool(0) {}

    long double findTezToTokenOutput(long double tez) const {
        return findOutput(tez, tez_pool, token_pool);
    }

    long double findTokenToTezOutput(long double token) const {
        return findOutput(token, token_pool, tez_pool);
    }

    virtual long double findOutput(long double aIn, long double aAmount, long double bAmount) const = 0;
};

struct DEX: public Exchange {

    DEX(const double fee_): Exchange() {
        fee = fee_;
    }

    long double findOutput(long double aIn, long double aAmount, long double bAmount) const override {
        if (aIn >= aAmount) {
            return -1;
        }
        const long double feeRatio = (100 - fee) / 100;
        const long double aInWithFee = aIn * feeRatio;
        return (aInWithFee * bAmount) / (aAmount + aInWithFee);
    }
};

struct CEX: public Exchange {
    CEX(const double fee_): Exchange() {
        fee = fee_;
    }

    long double findOutput(long double aIn, long double aAmount, long double bAmount) const override {
        if (aIn >= aAmount) {
            return -1;
        }
        const long double feeRatio = (100 - fee) / 100;
        const long double aInWithFee = aIn * feeRatio;
        return (aInWithFee / aAmount) * bAmount;
    }
};

struct Balance {
    long double token = 0;
    long double tez = 0;
};

struct Arbitrage {
    enum class ROUTE_TYPE {
        DEX_CEX,
        CEX_DEX
    };

    ROUTE_TYPE type;
    long double amount;
    long double profit;
};

std::optional<Arbitrage> find_arbitrage(const DEX* dex_ptr, const CEX* cex_ptr, const Balance* dex_balance_ptr, const Balance* cex_balance_ptr) {
    Arbitrage best_arbitrage{
        .type=Arbitrage::ROUTE_TYPE::DEX_CEX,
        .amount=0,
        .profit=-1
    };
    Arbitrage::ROUTE_TYPE type = Arbitrage::ROUTE_TYPE::DEX_CEX;
    std::array<const Exchange*, 2> exchanges_ptr = {dex_ptr, cex_ptr};
    std::array<const Balance*, 2> balances_ptr = {dex_balance_ptr, cex_balance_ptr};

    const auto findProfit = [&exchanges_ptr, &balances_ptr](const long double tezAmount) -> long double {
        if (tezAmount > balances_ptr[0]->tez) {
            return -1;
        }
        const long double tokenAmount = exchanges_ptr[0]->findTezToTokenOutput(tezAmount);
        if (tokenAmount < 0) {
            return -1;
        }
        if (tokenAmount > balances_ptr[1]->token) {
            return -1;
        }
        const long double finalTezAmount = exchanges_ptr[1]->findTokenToTezOutput(tokenAmount);
        if (finalTezAmount < 0) {
            return -1;
        }
        return finalTezAmount - tezAmount;
    };

    for (int _ = 0; _ < 2; _++) {
        long double l = 0, r = exchanges_ptr[0]->tez_pool;
        while (r - l > EPS) {
            const long double mid1 = l + (r - l) / 3;
            const long double mid2 = r - (r - l) / 3;

            const long double profit1 = findProfit(mid1);
            const long double profit2 = findProfit(mid2);

            if (profit1 > profit2 || profit1 < 0) {
                r = mid2;
                if (profit1 > best_arbitrage.profit) {
                    best_arbitrage.profit = profit1;
                    best_arbitrage.type = type;
                    best_arbitrage.amount = mid1;
                }
            } else {
                l = mid1;
                if (profit2 > best_arbitrage.profit) {
                    best_arbitrage.profit = profit2;
                    best_arbitrage.type = type;
                    best_arbitrage.amount = mid2;
                }
            }
        }

        std::swap(exchanges_ptr[0], exchanges_ptr[1]);
        std::swap(balances_ptr[0], balances_ptr[1]);
        type = Arbitrage::ROUTE_TYPE::CEX_DEX;
    }
    if (best_arbitrage.profit < 0) {
        return std::nullopt;
    }
    return best_arbitrage;
}

template<typename LineType>
class CSVReader {
public:
    CSVReader(const std::string &file_name) : file(file_name) {
        std::string line;
        std::getline(file, line);
        if (!file.eof()) {
            // read header of csv
            file >> next_line;
        }
    }

    const LineType &get_next_line() const {
        return next_line;
    }

    bool update_next_line() {
        if (!file.eof()) {
            file >> next_line;
            return true;
        }
        return false;
    }

    bool eof() const {
        return file.eof();
    }

private:
    std::ifstream file;
    LineType next_line;
};

//CSVReader<DEXTrade> dex_trades("../tezos/XTZ_kUSD.csv");
//CSVReader<CEXTrade> cex_trades("../binance/trades/XTZUSDT.csv");
    CSVReader<DEXTrade> dex_trades("../tezos/XTZ_tzBTC.csv");
    CSVReader<CEXTrade> cex_trades("../binance/trades/XTZBTC.csv");

void create_price_history() {
    DEX dex(0.3);
    CEX cex(0.1);

    auto cex_last_trade = cex_trades.get_next_line();
    auto dex_last_trade = dex_trades.get_next_line();
//    uint64_t last_trade_timestamp = 0;
    dex.token_pool = dex_last_trade.token_pool;
    dex.tez_pool = dex_last_trade.tez_pool;
    cex.token_pool = cex_last_trade.quote_qty;
    cex.tez_pool = cex_last_trade.qty;
    do {
        while (cex_last_trade.time <= dex_last_trade.time) {
            cex_last_trade = cex_trades.get_next_line();
            if (cex_last_trade.quote_qty > 0) {
                cex.token_pool = cex_last_trade.quote_qty;
                cex.tez_pool = cex_last_trade.qty;
            }
            if (!cex_trades.update_next_line()) {
                break;
            }
        }
        while (dex_last_trade.time <= cex_last_trade.time) {
            dex_last_trade = dex_trades.get_next_line();
            if (dex_last_trade.token_pool > 0) {
                dex.token_pool = dex_last_trade.token_pool;
                dex.tez_pool = dex_last_trade.tez_pool;
            }
            if (!dex_trades.update_next_line()) {
                break;
            }
        }
        if (dex_trades.eof() || cex_trades.eof()) {
            break;
        }
        std::cout << dex_last_trade.time << ",";
        std::cout << std::fixed << std::setprecision(4) << dex.token_pool / dex.tez_pool  << ",";
        std::cout << std::fixed << std::setprecision(4) << cex.token_pool / cex.tez_pool  << std::endl;
    } while (true);
}

void calculate_profit() {
    DEX dex(0.3);
    CEX cex(0.1);

    Balance dex_balance;
    dex_balance.tez = 1500;
    Balance cex_balance;
    cex_balance.token = 0.1;

    int cnt_arbitrages = 0;

    auto cex_last_trade = cex_trades.get_next_line();
    do {
        const auto &dex_next_trade = dex_trades.get_next_line();

        if (dex_next_trade.status != "applied") {
            continue;
        }

        dex.token_pool = dex_next_trade.token_pool;
        dex.tez_pool = dex_next_trade.tez_pool;

        bool updated = false;
        while (cex_trades.get_next_line().time <= dex_next_trade.time) {
            cex_last_trade = cex_trades.get_next_line();

            cex.token_pool = cex_last_trade.quote_qty * 10;
            cex.tez_pool = cex_last_trade.qty * 10;

            if (!cex_trades.update_next_line()) {
                break;
            }
            updated = true;
        }
        if (!updated) {
            continue;
        }
        if (cex_last_trade.time > dex_next_trade.time) {
            continue;
        }
        auto arbitrage_opt = find_arbitrage(&dex, &cex, &dex_balance, &cex_balance);
        if (!arbitrage_opt || arbitrage_opt->profit < 0.1) {
            continue;
        }
        if (arbitrage_opt->type == Arbitrage::ROUTE_TYPE::DEX_CEX) {
            auto tezInAmount = arbitrage_opt->amount;
            auto tokenAmount = dex.findTezToTokenOutput(tezInAmount);
            dex_balance.tez -= tezInAmount;
            dex_balance.token += tokenAmount;
            auto tezOutAmount = cex.findTokenToTezOutput(tokenAmount);
            cex_balance.token -= tokenAmount;
            cex_balance.tez += tezOutAmount;
        } else {
            auto tezInAmount = arbitrage_opt->amount;
            auto tokenAmount = cex.findTezToTokenOutput(tezInAmount);
            cex_balance.tez -= tezInAmount;
            cex_balance.token += tokenAmount;
            auto tezOutAmount = dex.findTokenToTezOutput(tokenAmount);
            dex_balance.token -= tokenAmount;
            dex_balance.tez += tezOutAmount;
        }
        std::cerr << "found arbitrage: " << ++cnt_arbitrages << std::endl;
        std::cerr << "\t" << "time: " << dex_next_trade.time << std::endl;
        std::cerr << std::fixed << std::setprecision(4) << "\t" << "profit: " <<  arbitrage_opt->profit << " " << "amount: " << arbitrage_opt->amount << std::endl;
        std::cerr << "\t" << "tez balance    : " << dex_balance.tez + cex_balance.tez << std::endl;
        std::cerr << "\t" << "dex balance tez: " << dex_balance.tez << std::endl;
        std::cerr << "\t" << "token balance    : " << dex_balance.token + cex_balance.token << std::endl;
        std::cerr << "\t" << "dex balance token: " << dex_balance.token << std::endl;

    } while (dex_trades.update_next_line());
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    calculate_profit();

    return 0;
}