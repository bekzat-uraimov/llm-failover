// TCP worker. Holds one Engine and streams tokens to whoever is connected (later: the router).
// Protocol is one JSON object per line in both directions; README.md lists the messages.
//
// Single threaded on purpose. Between tokens the worker polls the socket without blocking,
// so a PING or STOP is answered within one token's time even mid-generation. Replaying a
// history after RESUME works the same way, one position per loop, so the router's heartbeat
// keeps getting answers while a long recovery runs.

#include "../engine/engine.h"
#include "json.h"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <optional>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

namespace {

// A RESUME carries at most seq_len + 1 token ids, a few KB. Anything bigger is a bad client.
constexpr size_t MAX_LINE = 64 * 1024;

struct Session {
    std::string id;
    int64_t epoch;
    int steps;  // stop once this many forward passes have run, same meaning as llama2.c's -n
    bool resuming = false;
    std::chrono::steady_clock::time_point replay_start{};
};

std::string tokens_json(const std::vector<int>& tokens) {
    std::string out = "[";
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i) out += ",";
        out += std::to_string(tokens[i]);
    }
    return out + "]";
}

class Connection {
public:
    Connection(int fd, Engine& engine) : fd_(fd), engine_(engine) {}

    void serve() {
        std::string buf;
        char chunk[4096];
        while (alive_) {
            // Block while idle; while generating, just check for messages and get back to work
            pollfd p{fd_, POLLIN, 0};
            int ready = poll(&p, 1, session_ ? 0 : -1);
            if (ready < 0) {
                if (errno == EINTR) continue;
                return;
            }
            if (ready > 0) {
                ssize_t n = recv(fd_, chunk, sizeof chunk, 0);
                if (n <= 0) return;  // peer closed the connection; drop the session with it
                buf.append(chunk, n);
                size_t nl;
                while (alive_ && (nl = buf.find('\n')) != std::string::npos) {
                    std::string line = buf.substr(0, nl);
                    buf.erase(0, nl + 1);
                    handle(line);
                }
                if (buf.size() > MAX_LINE) {
                    error("line too long");
                    return;
                }
            }
            if (session_) step();
        }
    }

private:
    int fd_;
    Engine& engine_;
    std::optional<Session> session_;
    bool alive_ = true;

    void send_line(const std::string& line) {
        std::string data = line + "\n";
        size_t sent = 0;
        while (alive_ && sent < data.size()) {
            ssize_t n = send(fd_, data.data() + sent, data.size() - sent, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) alive_ = false;
            else sent += n;
        }
    }

    void error(const std::string& msg) {
        send_line("{\"type\":\"ERROR\",\"error\":" + json_quote(msg) + "}");
    }

    std::string header(const char* type, const Session& s) {
        return std::string("{\"type\":\"") + type + "\",\"session\":" + json_quote(s.id) +
               ",\"epoch\":" + std::to_string(s.epoch);
    }

    void handle(const std::string& line) {
        if (line.empty()) return;
        try {
            Json msg = Json::parse(line);
            const std::string& type = msg.at("type").as_string();
            if (type == "PING") {
                send_line("{\"type\":\"PONG\"}");
            } else if (type == "START") {
                start(msg);
            } else if (type == "RESUME") {
                resume(msg);
            } else if (type == "STOP") {
                stop(msg);
            } else {
                error("unknown message type: " + type);
            }
        } catch (const std::exception& e) {
            error(e.what());
        }
    }

    Session new_session(const Json& msg) {
        if (session_) throw std::invalid_argument("busy with session " + session_->id);
        int steps = engine_.seq_len();
        if (msg.has("steps")) {
            int64_t requested = msg.at("steps").as_int();
            if (requested < 1) throw std::invalid_argument("steps must be at least 1");
            if (requested < steps) steps = static_cast<int>(requested);
        }
        return {msg.at("session").as_string(), msg.at("epoch").as_int(), steps};
    }

    void start(const Json& msg) {
        Session s = new_session(msg);
        uint64_t seed = msg.at("seed").as_uint();
        if (seed == 0) throw std::invalid_argument("seed must be non-zero");
        std::vector<int> tokens = engine_.encode(msg.at("prompt").as_string());
        if (static_cast<int>(tokens.size()) > engine_.seq_len()) throw std::invalid_argument("prompt too long");
        // The prompt goes through the model in step(), one position per loop, like a resume
        engine_.begin_restore({tokens, static_cast<int>(tokens.size()), seed});
        session_ = s;
        // The router needs the prompt tokens to build its log; it has no tokenizer of its own
        send_line(header("STARTED", s) + ",\"tokens\":" + tokens_json(engine_.tokens()) + "}");
        std::fprintf(stderr, "session %s epoch %lld: started, %zu prompt tokens\n", s.id.c_str(),
                     static_cast<long long>(s.epoch), engine_.tokens().size());
    }

    void resume(const Json& msg) {
        Session s = new_session(msg);
        EngineState state;
        for (const Json& t : msg.at("tokens").items) {
            int64_t id = t.as_int();
            if (id < 0 || id > INT32_MAX) throw std::invalid_argument("token out of range");
            state.tokens.push_back(static_cast<int>(id));
        }
        state.pos = msg.has("pos") ? static_cast<int>(msg.at("pos").as_int()) : static_cast<int>(state.tokens.size());
        state.rng_state = msg.at("rng_state").as_uint();

        engine_.begin_restore(state);
        s.resuming = true;
        s.replay_start = std::chrono::steady_clock::now();
        session_ = s;
        if (!engine_.replaying()) resumed();
    }

    void resumed() {
        auto elapsed = std::chrono::steady_clock::now() - session_->replay_start;
        char ms[32];
        std::snprintf(ms, sizeof ms, "%.2f", std::chrono::duration<double, std::milli>(elapsed).count());
        int pos = static_cast<int>(engine_.tokens().size());
        send_line(header("RESUMED", *session_) + ",\"pos\":" + std::to_string(pos) + ",\"replay_ms\":" + ms + "}");
        std::fprintf(stderr, "session %s epoch %lld: resumed at pos %d, replay took %s ms\n", session_->id.c_str(),
                     static_cast<long long>(session_->epoch), pos, ms);
        session_->resuming = false;
    }

    void stop(const Json& msg) {
        const std::string& id = msg.at("session").as_string();
        if (session_ && session_->id == id) session_.reset();
        send_line("{\"type\":\"STOPPED\",\"session\":" + json_quote(id) + "}");
    }

    void finish(const char* reason) {
        send_line(header("DONE", *session_) + ",\"reason\":\"" + reason + "\"}");
        std::fprintf(stderr, "session %s epoch %lld: done (%s)\n", session_->id.c_str(),
                     static_cast<long long>(session_->epoch), reason);
        session_.reset();
    }

    void step() {
        if (engine_.replaying()) {
            engine_.replay_step();
            if (!engine_.replaying() && session_->resuming) resumed();
            return;
        }
        if (static_cast<int>(engine_.tokens().size()) - 1 >= session_->steps) {
            finish("steps");
            return;
        }
        int token = engine_.next_token();
        // BOS marks the end of a story, same stop rule as llama2.c
        if (token == Engine::BOS) {
            finish("eos");
            return;
        }
        EngineState s = engine_.save();
        send_line(header("TOKEN", *session_) + ",\"pos\":" + std::to_string(s.pos - 1) +
                  ",\"token\":" + std::to_string(token) + ",\"rng_state\":" + std::to_string(s.rng_state) + "}");
    }
};

int listen_on(const std::string& host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error(std::string("socket: ") + std::strerror(errno));
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) throw std::runtime_error("bad host: " + host);
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
        throw std::runtime_error(std::string("bind: ") + std::strerror(errno));
    }
    if (listen(fd, 8) != 0) throw std::runtime_error(std::string("listen: ") + std::strerror(errno));
    return fd;
}

int bound_port(int fd) {
    sockaddr_in addr{};
    socklen_t len = sizeof addr;
    getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
    return ntohs(addr.sin_port);
}

void usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s [--model PATH] [--tokenizer PATH] [--host ADDR] [--port N]\n"
                 "  defaults: models/stories15M.bin, models/tokenizer.bin, 127.0.0.1, 9001 (0 = any free port)\n",
                 prog);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model = "models/stories15M.bin";
    std::string tokenizer = "models/tokenizer.bin";
    std::string host = "127.0.0.1";
    int port = 9001;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (i + 1 >= argc) {
            usage(argv[0]);
            return 2;
        }
        if (arg == "--model") model = argv[++i];
        else if (arg == "--tokenizer") tokenizer = argv[++i];
        else if (arg == "--host") host = argv[++i];
        else if (arg == "--port") port = std::atoi(argv[++i]);
        else {
            usage(argv[0]);
            return 2;
        }
    }

    // A client that disconnects mid-send would otherwise kill the process with SIGPIPE
    std::signal(SIGPIPE, SIG_IGN);

    try {
        Engine engine(model, tokenizer);
        int listen_fd = listen_on(host, port);
        std::printf("worker listening on %s:%d\n", host.c_str(), bound_port(listen_fd));
        std::fflush(stdout);

        for (;;) {
            int fd = accept(listen_fd, nullptr, nullptr);
            if (fd < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error(std::string("accept: ") + std::strerror(errno));
            }
            // Tokens are tiny and latency matters more than packet count
            int one = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            std::fprintf(stderr, "connection opened\n");
            Connection(fd, engine).serve();
            close(fd);
            std::fprintf(stderr, "connection closed\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "worker: %s\n", e.what());
        return 1;
    }
}
