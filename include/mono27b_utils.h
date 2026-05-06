#pragma once

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace Mono27BUtils {

template<typename T>
inline bool read_pod(FILE *fp, T &out, std::string &error) {
    if (std::fread(&out, sizeof(T), 1, fp) != 1) {
        error = "failed to read POD";
        return false;
    }
    return true;
}

template<typename T>
inline bool read_at(FILE *fp, T &out, uint64_t off, std::string &error) {
    if (std::fseek(fp, (long)off, SEEK_SET) != 0) {
        error = "seek failed";
        return false;
    }
    return read_pod(fp, out, error);
}

inline bool read_bytes(FILE *fp, void *buf, size_t len, std::string &error) {
    if (std::fread(buf, len, 1, fp) != 1) {
        error = "read failed";
        return false;
    }
    return true;
}

inline bool read_string(FILE *fp, std::string &out, std::string &error) {
    uint32_t len = 0;
    if (!read_pod(fp, len, error)) return false;
    out.resize(len);
    if (len > 0 && !read_bytes(fp, out.data(), len, error)) return false;
    return true;
}

inline bool read_vector(FILE *fp, std::vector<std::string> &out, size_t count, std::string &error) {
    for (size_t i = 0; i < count; i++) {
        std::string s;
        if (!read_string(fp, s, error)) return false;
        out.push_back(std::move(s));
    }
    return true;
}

}
