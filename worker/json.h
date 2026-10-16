#pragma once

// Minimal JSON reader/writer for the worker's line protocol. Numbers keep their source
// text so a 64-bit rng_state round-trips exactly; a double only holds 53 bits.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    std::string text;  // string value, or a number's source text
    std::vector<Json> items;
    std::map<std::string, Json> fields;

    static Json parse(const std::string& s);

    const Json& at(const std::string& key) const {
        auto it = fields.find(key);
        if (type != Type::Object || it == fields.end()) throw std::invalid_argument("missing field: " + key);
        return it->second;
    }

    bool has(const std::string& key) const { return type == Type::Object && fields.count(key) > 0; }

    const std::string& as_string() const {
        if (type != Type::String) throw std::invalid_argument("expected a string");
        return text;
    }

    int64_t as_int() const {
        if (type != Type::Number || text.find_first_of(".eE") != std::string::npos) {
            throw std::invalid_argument("expected an integer");
        }
        errno = 0;
        long long v = std::strtoll(text.c_str(), nullptr, 10);
        if (errno == ERANGE) throw std::invalid_argument("integer out of range");
        return v;
    }

    uint64_t as_uint() const {
        if (type != Type::Number || text.find_first_of(".eE-") != std::string::npos) {
            throw std::invalid_argument("expected a non-negative integer");
        }
        errno = 0;
        unsigned long long v = std::strtoull(text.c_str(), nullptr, 10);
        if (errno == ERANGE) throw std::invalid_argument("integer out of range");
        return v;
    }
};

namespace json_detail {

class Parser {
public:
    explicit Parser(const std::string& s) : s_(s) {}

    Json parse_document() {
        Json v = value(0);
        skip_ws();
        if (i_ != s_.size()) fail("trailing characters");
        return v;
    }

private:
    const std::string& s_;
    size_t i_ = 0;

    [[noreturn]] void fail(const std::string& msg) {
        throw std::invalid_argument("bad JSON at " + std::to_string(i_) + ": " + msg);
    }

    void skip_ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) i_++;
    }

    char peek() {
        skip_ws();
        if (i_ >= s_.size()) fail("unexpected end");
        return s_[i_];
    }

    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        i_++;
    }

    bool consume(const char* word) {
        size_t n = std::char_traits<char>::length(word);
        if (s_.compare(i_, n, word) != 0) return false;
        i_ += n;
        return true;
    }

    Json value(int depth) {
        // Messages are flat; a depth cap keeps a hostile line from blowing the stack
        if (depth > 16) fail("nested too deep");
        Json v;
        char c = peek();
        if (c == '{') {
            v.type = Json::Type::Object;
            i_++;
            if (peek() == '}') {
                i_++;
                return v;
            }
            for (;;) {
                if (peek() != '"') fail("expected a key");
                std::string key = string();
                expect(':');
                v.fields[key] = value(depth + 1);
                if (peek() == ',') {
                    i_++;
                    continue;
                }
                expect('}');
                return v;
            }
        }
        if (c == '[') {
            v.type = Json::Type::Array;
            i_++;
            if (peek() == ']') {
                i_++;
                return v;
            }
            for (;;) {
                v.items.push_back(value(depth + 1));
                if (peek() == ',') {
                    i_++;
                    continue;
                }
                expect(']');
                return v;
            }
        }
        if (c == '"') {
            v.type = Json::Type::String;
            v.text = string();
            return v;
        }
        if (consume("true")) {
            v.type = Json::Type::Bool;
            v.boolean = true;
            return v;
        }
        if (consume("false")) {
            v.type = Json::Type::Bool;
            return v;
        }
        if (consume("null")) return v;
        if (c == '-' || (c >= '0' && c <= '9')) {
            v.type = Json::Type::Number;
            v.text = number();
            return v;
        }
        fail("unexpected character");
    }

    std::string number() {
        size_t start = i_;
        auto digits = [&] {
            size_t d = i_;
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') i_++;
            if (i_ == d) fail("expected a digit");
        };
        if (s_[i_] == '-') i_++;
        digits();
        if (i_ < s_.size() && s_[i_] == '.') {
            i_++;
            digits();
        }
        if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            i_++;
            if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) i_++;
            digits();
        }
        return s_.substr(start, i_ - start);
    }

    unsigned hex4() {
        if (i_ + 4 > s_.size()) fail("short \\u escape");
        unsigned v = 0;
        for (int k = 0; k < 4; k++) {
            char h = s_[i_++];
            v <<= 4;
            if (h >= '0' && h <= '9') v |= h - '0';
            else if (h >= 'a' && h <= 'f') v |= h - 'a' + 10;
            else if (h >= 'A' && h <= 'F') v |= h - 'A' + 10;
            else fail("bad \\u escape");
        }
        return v;
    }

    static void put_utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    std::string string() {
        expect('"');
        std::string out;
        for (;;) {
            if (i_ >= s_.size()) fail("unterminated string");
            char c = s_[i_++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
            if (c != '\\') {
                out += c;
                continue;
            }
            if (i_ >= s_.size()) fail("unterminated escape");
            char e = s_[i_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned cp = hex4();
                    // Characters outside the BMP arrive as a surrogate pair
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (!consume("\\u")) fail("lone surrogate");
                        unsigned lo = hex4();
                        if (lo < 0xDC00 || lo > 0xDFFF) fail("bad surrogate pair");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        fail("lone surrogate");
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }
};

}  // namespace json_detail

inline Json Json::parse(const std::string& s) {
    return json_detail::Parser(s).parse_document();
}

inline std::string json_quote(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out + "\"";
}
