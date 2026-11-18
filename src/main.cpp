// tc: command-line front end for the TC-Runtime codec.
//
//   tc gen        --dist gaussian --shape 4096,4096 [--seed 1] [--sigma 0.02] [--rho 0.9] [--dtype fp32] -o x.bin
//   tc compress   -i x.bin -o x.tcf --dtype fp32 --shape 4096,4096 [--flags sxr] [--block-size 65536]
//   tc decompress -i x.tcf -o x.out.bin
//   tc info       -i x.tcf
//   tc stats      -i x.tcf
//   tc analyze    -i x.bin --dtype fp32

#include "analyze.h"
#include "compressor.h"
#include "format.h"
#include "synthetic.h"
#include "tensor.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Args = std::map<std::string, std::string>;

Args parse_args(int argc, char** argv, int first) {
    Args args;
    for (int i = first; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("-", 0) != 0) throw std::invalid_argument("unexpected argument '" + key + "'");
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
        args[key] = argv[++i];
    }
    return args;
}

std::string required(const Args& args, const std::string& key) {
    auto it = args.find(key);
    if (it == args.end()) throw std::invalid_argument("missing required option " + key);
    return it->second;
}

std::string optional(const Args& args, const std::string& key, const std::string& fallback) {
    auto it = args.find(key);
    return it == args.end() ? fallback : it->second;
}

std::vector<size_t> parse_shape(const std::string& text) {
    std::vector<size_t> shape;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t comma = text.find(',', start);
        const std::string part = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (part.empty() || part.find_first_not_of("0123456789") != std::string::npos) {
            throw std::invalid_argument("bad --shape '" + text + "' (expected e.g. 4096,4096)");
        }
        shape.push_back(std::stoull(part));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return shape;
}

uint16_t parse_flags(const std::string& text) {
    if (text == "none") return 0;
    uint16_t flags = 0;
    for (char c : text) {
        switch (c) {
            case 's': flags |= tc::FLAG_SHUFFLE; break;
            case 'x': flags |= tc::FLAG_XOR; break;
            case 'r': flags |= tc::FLAG_RLE; break;
            default: throw std::invalid_argument("bad --flags '" + text + "' (letters s, x, r, or 'none')");
        }
    }
    return flags;
}

std::string flags_text(uint16_t flags) {
    std::string text;
    if (flags & tc::FLAG_SHUFFLE) text += "shuffle ";
    if (flags & tc::FLAG_XOR) text += "xor ";
    if (flags & tc::FLAG_RLE) text += "rle ";
    if (text.empty()) return "none";
    text.pop_back();
    return text;
}

std::string shape_text(const std::vector<size_t>& shape) {
    std::string text = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) text += ", ";
        text += std::to_string(shape[i]);
    }
    return text + "]";
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open '" + path + "' for reading");
    const std::streamsize size = in.tellg();
    in.seekg(0);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (size > 0 && !in.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("failed reading '" + path + "'");
    }
    return bytes;
}

void write_file(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open '" + path + "' for writing");
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("failed writing '" + path + "'");
}

void print_frame_summary(const tc::FrameStats& stats) {
    const tc::Tcf1Header& h = stats.header;
    std::printf("format:            TCF1 v%u\n", h.version);
    std::printf("dtype:             %s\n", tc::dtype_name(static_cast<tc::DataType>(h.dtype)));
    std::printf("shape:             %s\n", shape_text(tc::header_shape(h)).c_str());
    std::printf("stages:            %s\n", flags_text(h.flags).c_str());
    std::printf("block size:        %u bytes\n", h.block_size);
    std::printf("blocks:            %u\n", h.block_count);
    std::printf("uncompressed:      %llu bytes\n", static_cast<unsigned long long>(h.uncompressed_size));
    std::printf("frame:             %llu bytes (64 header + %llu payload)\n",
                static_cast<unsigned long long>(stats.frame_bytes),
                static_cast<unsigned long long>(h.payload_size));
}

int cmd_gen(const Args& args) {
    tc::SynthSpec spec;
    spec.dist = tc::parse_dist(required(args, "--dist"));
    spec.shape = parse_shape(required(args, "--shape"));
    spec.seed = std::stoull(optional(args, "--seed", "1"));
    spec.sigma = std::stod(optional(args, "--sigma", "0.02"));
    spec.rho = std::stod(optional(args, "--rho", "0.9"));
    const char* default_dtype = spec.dist == tc::Dist::QuantizedInt8 ? "int8" : "fp32";
    spec.dtype = tc::parse_dtype(optional(args, "--dtype", default_dtype));

    const tc::RawTensor tensor = tc::generate(spec);
    write_file(required(args, "-o"), tensor.data);
    std::printf("wrote %zu bytes: %s %s %s seed=%llu\n", tensor.data.size(), tc::dist_name(spec.dist),
                tc::dtype_name(spec.dtype), shape_text(spec.shape).c_str(),
                static_cast<unsigned long long>(spec.seed));
    return 0;
}

int cmd_compress(const Args& args) {
    tc::RawTensor tensor;
    tensor.data = read_file(required(args, "-i"));
    tensor.dtype = tc::parse_dtype(required(args, "--dtype"));
    tensor.shape = parse_shape(required(args, "--shape"));

    tc::CompressOptions options;
    options.flags = parse_flags(optional(args, "--flags", "sxr"));
    options.block_size = static_cast<uint32_t>(std::stoul(optional(args, "--block-size", "65536")));

    tc::TensorCompressor compressor(options);
    const tc::CompressedTensor compressed = compressor.compress(tensor);
    write_file(required(args, "-o"), compressed.compressed_data);
    std::printf("%zu -> %zu bytes (ratio %.4fx)\n", tensor.data.size(), compressed.compressed_data.size(),
                tc::inspect_frame(compressed.compressed_data).ratio);
    return 0;
}

int cmd_decompress(const Args& args) {
    const tc::CompressedTensor compressed = tc::compressed_from_frame(read_file(required(args, "-i")));
    tc::TensorCompressor compressor;
    const tc::RawTensor tensor = compressor.decompress(compressed);
    write_file(required(args, "-o"), tensor.data);
    std::printf("%zu -> %zu bytes\n", compressed.compressed_data.size(), tensor.data.size());
    return 0;
}

int cmd_info(const Args& args) {
    const std::vector<uint8_t> frame = read_file(required(args, "-i"));
    print_frame_summary(tc::inspect_frame(frame));
    return 0;
}

int cmd_stats(const Args& args) {
    const std::vector<uint8_t> frame = read_file(required(args, "-i"));
    const tc::FrameStats stats = tc::inspect_frame(frame);
    print_frame_summary(stats);
    const uint32_t blocks = stats.header.block_count;
    std::printf("ratio:             %.4fx\n", stats.ratio);
    std::printf("encoded blocks:    %u\n", stats.encoded_blocks);
    std::printf("stored-raw blocks: %u (%.1f%%)\n", stats.stored_blocks,
                blocks == 0 ? 0.0 : 100.0 * stats.stored_blocks / blocks);
    return 0;
}

int cmd_analyze(const Args& args) {
    tc::RawTensor tensor;
    tensor.data = read_file(required(args, "-i"));
    tensor.dtype = tc::parse_dtype(required(args, "--dtype"));
    tensor.shape = {tensor.data.size() / tc::element_size(tensor.dtype)};
    if (tensor.data.size() % tc::element_size(tensor.dtype) != 0) {
        throw std::invalid_argument("file size is not a multiple of the element size");
    }

    const tc::AnalysisReport report = tc::analyze(tensor);
    std::printf("elements: %llu x %zu bytes (%s)\n", static_cast<unsigned long long>(report.n_elems),
                report.elem_size, tc::dtype_name(tensor.dtype));
    std::printf("\nbyte plane  H(bits/byte)  H after xor  xor+rle size\n");
    for (size_t j = 0; j < report.planes.size(); ++j) {
        const tc::PlaneStats& p = report.planes[j];
        std::printf("  byte %zu      %6.3f        %6.3f      %6.2f%%\n", j, p.entropy_bits, p.xor_entropy_bits,
                    p.bytes == 0 ? 0.0 : 100.0 * static_cast<double>(p.rle_bytes) / static_cast<double>(p.bytes));
    }
    if (report.exponent_field_bits != 0) {
        std::printf("\nsign field:     %.3f of 1 bit\n", report.sign_entropy_bits);
        std::printf("exponent field: %.3f of %d bits\n", report.exponent_entropy_bits, report.exponent_field_bits);
    }
    std::printf("\norder-0 bound, per-plane static code:      %.4fx\n", report.order0_ratio);
    std::printf("order-0 bound after xor-delta:             %.4fx\n", report.xor_order0_ratio);
    std::printf("per-plane xor + rle (what TC-Runtime gets): %.4fx\n", report.plane_rle_ratio);
    return 0;
}

void usage() {
    std::fputs(
        "usage: tc <command> [options]\n"
        "  gen        --dist zeros|random_bits|gaussian|ar1_gaussian|shared_exponent|quantized_int8\n"
        "             --shape D0,D1,... [--seed N] [--sigma S] [--rho R] [--dtype fp32|fp16|int8] -o FILE\n"
        "  compress   -i RAW -o FRAME --dtype fp32|fp16|int8 --shape D0,D1,...\n"
        "             [--flags sxr|none] [--block-size BYTES]\n"
        "  decompress -i FRAME -o RAW\n"
        "  info       -i FRAME\n"
        "  stats      -i FRAME\n"
        "  analyze    -i RAW --dtype fp32|fp16|int8\n",
        stderr);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 2;
    }
    const std::string command = argv[1];
    try {
        const Args args = parse_args(argc, argv, 2);
        if (command == "gen") return cmd_gen(args);
        if (command == "compress") return cmd_compress(args);
        if (command == "decompress") return cmd_decompress(args);
        if (command == "info") return cmd_info(args);
        if (command == "stats") return cmd_stats(args);
        if (command == "analyze") return cmd_analyze(args);
        usage();
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tc %s: %s\n", command.c_str(), e.what());
        return 1;
    }
}
