#include "mono27b_tokenizer.h"

#include <algorithm>
#include <climits>
#include <cstdio>

bool Mono27BTokenizer::load_from_blob(const std::string & blob_path,
                                      const Mono27BBlobHeader & header,
                                      std::string & error) {
    if (header.tokenizer_offset == 0 || header.tokenizer_bytes == 0) {
        error = "missing tokenizer section in blob";
        return false;
    }
    if (header.tokenizer_offset + header.tokenizer_bytes > header.file_bytes) {
        error = "invalid tokenizer section in blob";
        return false;
    }
    if (!mono27b_read_tokenizer_section(blob_path, tokenizer_, error)) {
        return false;
    }

    vocab_.clear();
    str_to_id_.clear();
    merge_rank_.clear();
    special_tokens_.clear();

    std::FILE * fp = std::fopen(blob_path.c_str(), "rb");
    if (!fp) {
        error = "missing blob: " + blob_path;
        return false;
    }
    const uint64_t base = header.tokenizer_offset + sizeof(Mono27BTokenizerSection);
    if (std::fseek(fp, static_cast<long>(base), SEEK_SET) != 0) {
        std::fclose(fp);
        error = "failed to seek tokenizer payload";
        return false;
    }

    vocab_.reserve(tokenizer_.vocab_size);
    for (uint32_t i = 0; i < tokenizer_.vocab_size; ++i) {
        uint32_t len = 0;
        if (std::fread(&len, sizeof(len), 1, fp) != 1) {
            std::fclose(fp);
            error = "failed to read tokenizer token length";
            return false;
        }
        std::string token;
        token.resize(len);
        if (len > 0 && std::fread(token.data(), 1, len, fp) != len) {
            std::fclose(fp);
            error = "failed to read tokenizer token payload";
            return false;
        }
        str_to_id_[token] = static_cast<int32_t>(i);
        vocab_.push_back(std::move(token));
    }

    for (uint32_t i = 0; i < tokenizer_.merges_count; ++i) {
        uint32_t len = 0;
        if (std::fread(&len, sizeof(len), 1, fp) != 1) {
            std::fclose(fp);
            error = "failed to read tokenizer merge length";
            return false;
        }
        std::string merge;
        merge.resize(len);
        if (len > 0 && std::fread(merge.data(), 1, len, fp) != len) {
            std::fclose(fp);
            error = "failed to read tokenizer merge payload";
            return false;
        }
        merge_rank_[merge] = static_cast<int>(i);
    }

    std::fclose(fp);
    for (int32_t i = 0; i < static_cast<int32_t>(vocab_.size()); ++i) {
        const std::string & token = vocab_[static_cast<size_t>(i)];
        if (token.size() >= 4 && token.rfind("<|", 0) == 0) {
            special_tokens_.emplace_back(token, i);
        }
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const auto & a, const auto & b) {
                  if (a.first.size() != b.first.size()) return a.first.size() > b.first.size();
                  return a.second < b.second;
              });
    init_byte_tables();
    return vocab_.size() == tokenizer_.vocab_size;
}

void Mono27BTokenizer::init_byte_tables() {
    std::vector<int> bs;
    std::vector<int> cs;
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) { bs.push_back(b); cs.push_back(b); }
    for (int b = 0xA1; b <= 0xAC; ++b) { bs.push_back(b); cs.push_back(b); }
    for (int b = 0xAE; b <= 0xFF; ++b) { bs.push_back(b); cs.push_back(b); }
    int c = 256;
    for (int b = 0; b < 256; ++b) {
        bool in_bs = false;
        for (int v : bs) {
            if (v == b) {
                in_bs = true;
                break;
            }
        }
        if (!in_bs) {
            bs.push_back(b);
            cs.push_back(c++);
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        const int cp = cs[i];
        std::string u8;
        if (cp < 0x80) {
            u8 += static_cast<char>(cp);
        } else if (cp < 0x800) {
            u8 += static_cast<char>(0xC0 | (cp >> 6));
            u8 += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            u8 += static_cast<char>(0xE0 | (cp >> 12));
            u8 += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            u8 += static_cast<char>(0x80 | (cp & 0x3F));
        }
        byte_to_unicode_[static_cast<size_t>(bs[i])] = u8;
        unicode_to_byte_[u8] = static_cast<uint8_t>(bs[i]);
    }
}

std::vector<int32_t> Mono27BTokenizer::encode(const std::string & text) const {
    std::vector<std::string> chunks;
    size_t i = 0;
    while (i < text.size()) {
        if (text.compare(i, 12, "<|im_start|>") == 0) {
            chunks.push_back("<");
            chunks.push_back("|");
            chunks.push_back("im");
            chunks.push_back("_start");
            chunks.push_back("|");
            chunks.push_back(">");
            i += 12;
            continue;
        }
        if (text.compare(i, 10, "<|im_end|>") == 0) {
            chunks.push_back("<");
            chunks.push_back("|");
            chunks.push_back("im");
            chunks.push_back("_end");
            chunks.push_back("|");
            chunks.push_back(">");
            i += 10;
            continue;
        }

        unsigned char c = static_cast<unsigned char>(text[i]);
        std::string chunk;
        if (c == ' ') {
            chunk += ' ';
            i++;
            if (i >= text.size()) {
                chunks.push_back(chunk);
                break;
            }
            c = static_cast<unsigned char>(text[i]);
        }

        if (c == '\'' && i + 1 < text.size()) {
            const char n = text[i + 1];
            if (n == 's' || n == 't' || n == 'm' || n == 'd') {
                chunk += text[i];
                chunk += text[i + 1];
                i += 2;
                chunks.push_back(chunk);
                continue;
            }
            if (i + 2 < text.size() && text[i + 1] == 'l' && text[i + 2] == 'l') {
                chunk += text[i];
                chunk += "ll";
                i += 3;
                chunks.push_back(chunk);
                continue;
            }
            if (i + 2 < text.size() && text[i + 1] == 'v' && text[i + 2] == 'e') {
                chunk += text[i];
                chunk += "ve";
                i += 3;
                chunks.push_back(chunk);
                continue;
            }
            if (i + 2 < text.size() && text[i + 1] == 'r' && text[i + 2] == 'e') {
                chunk += text[i];
                chunk += "re";
                i += 3;
                chunks.push_back(chunk);
                continue;
            }
        }

        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80) {
            while (i < text.size()) {
                c = static_cast<unsigned char>(text[i]);
                if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80) {
                    chunk += static_cast<char>(c);
                    i++;
                } else {
                    break;
                }
            }
        } else if (c >= '0' && c <= '9') {
            while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
                chunk += text[i];
                i++;
            }
        } else if (c != ' ' && c != '\n' && c != '\t' && c != '\r') {
            while (i < text.size()) {
                c = static_cast<unsigned char>(text[i]);
                if (c == ' ' || c == '\n' || c == '\t' || c == '\r' ||
                    (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9')) {
                    break;
                }
                chunk += static_cast<char>(c);
                i++;
            }
        } else {
            while (i < text.size()) {
                c = static_cast<unsigned char>(text[i]);
                if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
                    chunk += static_cast<char>(c);
                    i++;
                } else {
                    break;
                }
            }
        }

        if (!chunk.empty()) {
            chunks.push_back(chunk);
        }
    }

    std::vector<int32_t> ids;
    for (const auto & chunk : chunks) {
        std::string uni;
        for (unsigned char ch : chunk) {
            uni += byte_to_unicode_[ch];
        }
        const auto chunk_ids = bpe(uni);
        ids.insert(ids.end(), chunk_ids.begin(), chunk_ids.end());
    }
    return ids;
}

std::vector<int32_t> Mono27BTokenizer::bpe(const std::string & word) const {
    std::vector<std::string> syms;
    size_t i = 0;
    while (i < word.size()) {
        const size_t cp_len = utf8_len(word[i]);
        syms.push_back(word.substr(i, cp_len));
        i += cp_len;
    }
    if (syms.empty()) {
        return {};
    }

    std::string whole;
    for (const auto & s : syms) {
        whole += s;
    }
    const auto it = str_to_id_.find(whole);
    if (it != str_to_id_.end()) {
        return {it->second};
    }

    while (syms.size() > 1) {
        int best_rank = INT_MAX;
        int best_idx = -1;
        for (size_t j = 0; j + 1 < syms.size(); ++j) {
            const std::string pair = syms[j] + " " + syms[j + 1];
            const auto mit = merge_rank_.find(pair);
            if (mit != merge_rank_.end() && mit->second < best_rank) {
                best_rank = mit->second;
                best_idx = static_cast<int>(j);
            }
        }
        if (best_idx < 0) {
            break;
        }
        syms[static_cast<size_t>(best_idx)] += syms[static_cast<size_t>(best_idx + 1)];
        syms.erase(syms.begin() + best_idx + 1);
    }

    std::vector<int32_t> ids;
    for (const auto & s : syms) {
        const auto it2 = str_to_id_.find(s);
        if (it2 != str_to_id_.end()) {
            ids.push_back(it2->second);
        } else {
            for (char ch : s) {
                const std::string byte_str = byte_to_unicode_[static_cast<unsigned char>(ch)];
                const auto it3 = str_to_id_.find(byte_str);
                if (it3 != str_to_id_.end()) {
                    ids.push_back(it3->second);
                }
            }
        }
    }
    return ids;
}

std::string Mono27BTokenizer::decode(const std::vector<int32_t> & ids) const {
    std::string uni;
    for (int32_t id : ids) {
        if (should_skip_token(id)) {
            continue;
        }
        uni += vocab_[static_cast<size_t>(id)];
    }

    std::string out;
    size_t i = 0;
    while (i < uni.size()) {
        const size_t cp_len = utf8_len(uni[i]);
        const std::string ch = uni.substr(i, cp_len);
        i += cp_len;
        const auto it = unicode_to_byte_.find(ch);
        if (it != unicode_to_byte_.end()) {
            out += static_cast<char>(it->second);
        } else {
            out += ch;
        }
    }
    return out;
}

std::string Mono27BTokenizer::decode_one(int32_t id) const {
    if (should_skip_token(id)) {
        return "";
    }
    std::string uni = vocab_[static_cast<size_t>(id)];
    std::string out;
    size_t i = 0;
    while (i < uni.size()) {
        const size_t cp_len = utf8_len(uni[i]);
        const std::string ch = uni.substr(i, cp_len);
        i += cp_len;
        const auto it = unicode_to_byte_.find(ch);
        if (it != unicode_to_byte_.end()) {
            out += static_cast<char>(it->second);
        } else {
            out += ch;
        }
    }
    return out;
}

bool Mono27BTokenizer::is_terminal(int32_t id) const {
    if (static_cast<uint32_t>(id) == tokenizer_.im_end_id || static_cast<uint32_t>(id) == tokenizer_.eos_id) {
        return true;
    }
    if (id < 0 || static_cast<size_t>(id) >= vocab_.size()) {
        return false;
    }
    const std::string & tok = vocab_[static_cast<size_t>(id)];
    return tok == "<|endoftext|>" || tok == "</s>";
}

bool Mono27BTokenizer::should_skip_token(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= vocab_.size()) {
        return true;
    }
    if (static_cast<uint32_t>(id) == tokenizer_.im_start_id ||
        static_cast<uint32_t>(id) == tokenizer_.im_end_id ||
        static_cast<uint32_t>(id) == tokenizer_.eos_id) {
        return true;
    }
    const std::string & tok = vocab_[static_cast<size_t>(id)];
    return tok == "<|endoftext|>" || tok == "</s>";
}

size_t Mono27BTokenizer::utf8_len(char c) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (uc < 0x80) return 1;
    if ((uc & 0xE0) == 0xC0) return 2;
    if ((uc & 0xF0) == 0xE0) return 3;
    if ((uc & 0xF8) == 0xF0) return 4;
    return 1;
}
