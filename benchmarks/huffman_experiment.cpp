#include "huffman_experiment.h"

#include "codec/byte_shuffle.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tc::huffman {
namespace {

struct Node {
    int symbol = -1;    // 0..255 for leaves, -1 for internals
    uint64_t freq = 0;
    int left = -1;
    int right = -1;
};

struct Code {
    uint32_t bits = 0;
    uint8_t length = 0;
};

void build_codes(const std::vector<Node>& nodes, int idx, uint32_t bits, uint8_t len,
                 std::array<Code, 256>& out) {
    if (idx < 0) return;
    if (nodes[idx].left < 0 && nodes[idx].right < 0) {
        if (nodes[idx].symbol >= 0) {
            out[static_cast<size_t>(nodes[idx].symbol)] = {bits, len == 0 ? uint8_t{1} : len};
        }
        return;
    }
    build_codes(nodes, nodes[idx].left, bits << 1, static_cast<uint8_t>(len + 1), out);
    build_codes(nodes, nodes[idx].right, (bits << 1) | 1, static_cast<uint8_t>(len + 1), out);
}

std::array<Code, 256> build_table(const std::array<uint64_t, 256>& hist) {
    std::vector<Node> nodes;
    nodes.reserve(512);
    struct Item {
        uint64_t freq;
        int idx;
        bool operator>(const Item& o) const { return freq > o.freq; }
    };
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
    for (int s = 0; s < 256; ++s) {
        if (hist[static_cast<size_t>(s)] == 0) continue;
        Node n;
        n.symbol = s;
        n.freq = hist[static_cast<size_t>(s)];
        const int idx = static_cast<int>(nodes.size());
        nodes.push_back(n);
        pq.push({n.freq, idx});
    }
    std::array<Code, 256> codes{};
    if (pq.empty()) return codes;
    if (pq.size() == 1) {
        codes[static_cast<size_t>(nodes[pq.top().idx].symbol)] = {0, 1};
        return codes;
    }
    while (pq.size() > 1) {
        const Item a = pq.top();
        pq.pop();
        const Item b = pq.top();
        pq.pop();
        Node parent;
        parent.freq = a.freq + b.freq;
        parent.left = a.idx;
        parent.right = b.idx;
        const int idx = static_cast<int>(nodes.size());
        nodes.push_back(parent);
        pq.push({parent.freq, idx});
    }
    build_codes(nodes, pq.top().idx, 0, 0, codes);
    return codes;
}

}  // namespace

HuffmanResult compress_exponent_plane(std::span<const uint8_t> fp32_bytes) {
    if (fp32_bytes.size() % 4 != 0) {
        throw std::invalid_argument("compress_exponent_plane: size must be a multiple of 4");
    }
    HuffmanResult result;
    result.input_bytes = fp32_bytes.size();
    const size_t n = fp32_bytes.size() / 4;
    result.exponent_bytes = n;

    std::vector<uint8_t> shuffled(fp32_bytes.size());
    codec::byte_shuffle_scalar(fp32_bytes.data(), shuffled.data(), fp32_bytes.size(), 4);
    const uint8_t* plane3 = shuffled.data() + 3 * n;

    std::array<uint64_t, 256> hist{};
    for (size_t i = 0; i < n; ++i) ++hist[plane3[i]];
    const auto codes = build_table(hist);

    // Bit-pack the coded plane. Table itself: 256 * (code bits as u16 length+bits) ≈ 512 bytes.
    std::vector<uint8_t> bitstream;
    bitstream.reserve(n);
    uint64_t acc = 0;
    int acc_bits = 0;
    auto flush_byte = [&]() {
        bitstream.push_back(static_cast<uint8_t>(acc & 0xFF));
        acc >>= 8;
        acc_bits -= 8;
    };
    for (size_t i = 0; i < n; ++i) {
        const Code c = codes[plane3[i]];
        acc |= static_cast<uint64_t>(c.bits) << acc_bits;
        acc_bits += c.length;
        while (acc_bits >= 8) flush_byte();
    }
    if (acc_bits > 0) {
        bitstream.push_back(static_cast<uint8_t>(acc & 0xFF));
    }
    result.encoded_exponent_bytes = bitstream.size();

    // Toy frame: 4-byte magic + 256*2 code lengths + 256*4 code bits + raw planes 0..2 + bitstream.
    // Count only payload sizes that matter for the ratio claim.
    const size_t table_bytes = 256 * (2 + 4);  // length u16 + bits u32 per symbol
    const size_t raw_planes = 3 * n;
    result.total_frame_bytes = 4 + table_bytes + raw_planes + bitstream.size();
    result.ratio = result.total_frame_bytes == 0
                       ? 0.0
                       : static_cast<double>(result.input_bytes) / static_cast<double>(result.total_frame_bytes);
    return result;
}

}  // namespace tc::huffman
